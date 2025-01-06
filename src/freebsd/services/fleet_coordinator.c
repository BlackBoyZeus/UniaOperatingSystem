/**
 * @file fleet_coordinator.c
 * @version 1.0.0
 * @brief Implementation of fleet coordination service with power-aware operations
 * @copyright TALD UNIA Platform
 */

#include "fleet_coordinator.h"
#include <sys/types.h>
#include <sys/param.h>
#include <sys/mutex.h>
#include <string.h>
#include <errno.h>

// Version and system constants
static const char* FLEET_COORDINATOR_VERSION_STR = "1.0.0";
static const uint32_t FLEET_SYNC_INTERVAL_MS = 50;
static const uint32_t MAX_FLEET_RETRIES = 3;
static const uint32_t POWER_STATE_CHECK_INTERVAL_MS = 1000;
static const uint8_t BATTERY_THRESHOLD_LOW = 20;

// Global state
static fleet_coordinator_t* g_fleet_coordinator = NULL;
static struct mtx g_fleet_mutex;

// Forward declarations of internal functions
static int initialize_mesh_network(mesh_network_config_t* config);
static int setup_fleet_monitoring(fleet_coordinator_config_t* config);
static void monitor_fleet_power_state(fleet_handle_t* fleet);
static void update_fleet_metrics(fleet_handle_t* fleet);
static int validate_fleet_config(fleet_coordinator_config_t* config);

/**
 * @brief Initialize the fleet coordination service
 * @param config Service configuration parameters
 * @return 0 on success, error code on failure
 */
__attribute__((init))
int fleet_coordinator_init(fleet_coordinator_config_t* config) {
    int error;

    // Validate input parameters
    if (!config) {
        return EINVAL;
    }

    error = validate_fleet_config(config);
    if (error != 0) {
        return error;
    }

    // Initialize global mutex
    mtx_init(&g_fleet_mutex, "fleet_coordinator_mutex", NULL, MTX_DEF);

    // Allocate global coordinator
    g_fleet_coordinator = malloc(sizeof(fleet_coordinator_t));
    if (!g_fleet_coordinator) {
        mtx_destroy(&g_fleet_mutex);
        return ENOMEM;
    }
    memset(g_fleet_coordinator, 0, sizeof(fleet_coordinator_t));

    // Initialize mesh network subsystem
    error = initialize_mesh_network(&config->mesh_config);
    if (error != 0) {
        free(g_fleet_coordinator);
        mtx_destroy(&g_fleet_mutex);
        return error;
    }

    // Setup fleet monitoring
    error = setup_fleet_monitoring(config);
    if (error != 0) {
        free(g_fleet_coordinator);
        mtx_destroy(&g_fleet_mutex);
        return error;
    }

    return 0;
}

/**
 * @brief Update fleet power state based on battery and activity
 * @param fleet Fleet handle
 * @param new_state New power state
 * @return 0 on success, error code on failure
 */
static int fleet_coordinator_power_state_update(fleet_handle_t* fleet, power_state_t new_state) {
    int error = 0;

    if (!fleet) {
        return EINVAL;
    }

    mtx_lock(&g_fleet_mutex);

    // Update mesh network power configuration
    mesh_fleet_t* mesh_fleet = fleet->mesh_fleet;
    if (mesh_fleet) {
        error = mesh_set_fleet_priority(mesh_fleet, new_state);
        if (error != 0) {
            mtx_unlock(&g_fleet_mutex);
            return error;
        }
    }

    // Update synchronization intervals based on power state
    uint32_t sync_interval = FLEET_SYNC_INTERVAL_MS;
    switch (new_state) {
        case POWER_STATE_LOW:
            sync_interval *= 2; // Reduce sync frequency for power saving
            break;
        case POWER_STATE_HIGH:
            sync_interval = (uint32_t)(FLEET_SYNC_INTERVAL_MS * 0.75); // Increase sync frequency
            break;
    }

    // Update fleet configuration
    fleet->sync_latency_ms = sync_interval;
    fleet->power_state = new_state;

    // Notify fleet members of power state change
    if (mesh_fleet && mesh_fleet->num_devices > 0) {
        for (uint32_t i = 0; i < mesh_fleet->num_devices; i++) {
            mesh_peer_t* peer = &mesh_fleet->devices[i];
            if (peer && peer->state == MESH_STATE_CONNECTED) {
                mesh_protocol_send(peer, &new_state, sizeof(new_state));
            }
        }
    }

    mtx_unlock(&g_fleet_mutex);
    return 0;
}

/**
 * @brief Monitor fleet power consumption and optimize states
 * @param fleet Fleet handle
 */
static void monitor_fleet_power_state(fleet_handle_t* fleet) {
    if (!fleet || !fleet->mesh_fleet) {
        return;
    }

    // Calculate average battery level across fleet
    float total_battery = 0;
    uint32_t device_count = 0;
    mesh_fleet_t* mesh_fleet = fleet->mesh_fleet;

    for (uint32_t i = 0; i < mesh_fleet->num_devices; i++) {
        mesh_peer_t* peer = &mesh_fleet->devices[i];
        if (peer && peer->state == MESH_STATE_CONNECTED) {
            // Get battery level from peer metrics
            uint8_t battery_level = (peer->connection_quality >> 24) & 0xFF;
            total_battery += battery_level;
            device_count++;
        }
    }

    if (device_count > 0) {
        float avg_battery = total_battery / device_count;
        power_state_t new_state;

        // Determine optimal power state based on battery levels
        if (avg_battery <= BATTERY_THRESHOLD_LOW) {
            new_state = POWER_STATE_LOW;
        } else if (avg_battery >= 75) {
            new_state = POWER_STATE_HIGH;
        } else {
            new_state = POWER_STATE_BALANCED;
        }

        // Update power state if needed
        if (new_state != fleet->power_state) {
            fleet_coordinator_power_state_update(fleet, new_state);
        }
    }
}

/**
 * @brief Update fleet performance metrics
 * @param fleet Fleet handle
 */
static void update_fleet_metrics(fleet_handle_t* fleet) {
    if (!fleet || !fleet->mesh_fleet) {
        return;
    }

    mesh_fleet_t* mesh_fleet = fleet->mesh_fleet;
    monitor_stats_t* stats = &fleet->stats;

    // Update latency metrics
    stats->avg_sync_latency_ms = mesh_fleet->avg_latency_ms;
    stats->active_devices = mesh_fleet->num_devices;

    // Track sync failures
    for (uint32_t i = 0; i < mesh_fleet->num_devices; i++) {
        mesh_peer_t* peer = &mesh_fleet->devices[i];
        if (peer && peer->state == MESH_STATE_ERROR) {
            stats->sync_failures++;
        }
    }

    stats->last_update_time = mesh_fleet->last_heartbeat_ms;
}

/**
 * @brief Validate fleet coordinator configuration
 * @param config Configuration to validate
 * @return 0 if valid, error code otherwise
 */
static int validate_fleet_config(fleet_coordinator_config_t* config) {
    if (config->version != 1) {
        return EINVAL;
    }

    if (config->max_fleets > MESH_MAX_FLEETS || 
        config->max_fleets == 0) {
        return EINVAL;
    }

    if (config->sync_interval_ms < 10 || 
        config->sync_interval_ms > 1000) {
        return EINVAL;
    }

    if (config->recovery_timeout_ms < 1000 || 
        config->recovery_timeout_ms > 60000) {
        return EINVAL;
    }

    return 0;
}

/**
 * @brief Initialize mesh network subsystem
 * @param config Mesh network configuration
 * @return 0 on success, error code on failure
 */
static int initialize_mesh_network(mesh_network_config_t* config) {
    if (!config) {
        return EINVAL;
    }

    // Configure mesh network parameters
    config->version = MESH_NETWORK_VERSION;
    config->max_fleets = MESH_MAX_FLEETS;
    config->devices_per_fleet = MESH_MAX_DEVICES_PER_FLEET;
    config->target_latency_ms = MESH_TARGET_LATENCY_MS;

    // Initialize mesh network subsystem
    return mesh_network_init(config, M_DEVBUF);
}

/**
 * @brief Setup fleet monitoring services
 * @param config Fleet coordinator configuration
 * @return 0 on success, error code on failure
 */
static int setup_fleet_monitoring(fleet_coordinator_config_t* config) {
    if (!config || !g_fleet_coordinator) {
        return EINVAL;
    }

    // Initialize monitoring configuration
    g_fleet_coordinator->metrics = calloc(1, sizeof(metrics_collector_t));
    if (!g_fleet_coordinator->metrics) {
        return ENOMEM;
    }

    // Configure monitoring intervals
    g_fleet_coordinator->metrics->power_check_interval = POWER_STATE_CHECK_INTERVAL_MS;
    g_fleet_coordinator->metrics->sync_check_interval = config->sync_interval_ms;

    return 0;
}