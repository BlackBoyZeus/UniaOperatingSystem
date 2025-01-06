/**
 * @file fleet_coordinator.h
 * @version 1.0.0
 * @brief Service-level fleet coordination system for TALD UNIA platform with 32-device support
 * @copyright TALD UNIA Platform
 */

#ifndef TALD_FLEET_COORDINATOR_H
#define TALD_FLEET_COORDINATOR_H

// External dependencies with versions
#include <sys/types.h>  // FreeBSD 9.0
#include <sys/param.h>  // FreeBSD 9.0

// Internal dependencies
#include "../kernel/mesh_network.h"  // v1.0
#include "game_engine.h"            // v1.0.0

#include <atomic>
#include <mutex>
#include <string>

// Version and system constants
#define FLEET_COORDINATOR_VERSION "1.0.0"
#define MAX_FLEET_SIZE 32
#define MAX_FLEET_NAME_LENGTH 64
#define TARGET_SYNC_INTERVAL_MS 50
#define MIN_BATTERY_THRESHOLD 20
#define MAX_RECOVERY_ATTEMPTS 3
#define MONITORING_INTERVAL_MS 1000

namespace tald {
namespace fleet {

/**
 * @brief Power state enumeration for fleet coordination
 */
enum class FleetPowerState {
    LOW_POWER,
    BALANCED,
    HIGH_PERFORMANCE
};

/**
 * @brief Recovery state tracking for fleet resilience
 */
struct recovery_state_t {
    uint32_t attempt_count{0};
    uint64_t last_attempt_time{0};
    bool recovery_in_progress{false};
    std::string last_error;
};

/**
 * @brief Performance monitoring statistics
 */
struct monitor_stats_t {
    float avg_sync_latency_ms{0.0f};
    float avg_battery_level{0.0f};
    uint32_t active_devices{0};
    uint32_t sync_failures{0};
    uint64_t last_update_time{0};
};

/**
 * @brief Thread-safe configuration for fleet coordinator
 */
struct fleet_coordinator_config_t {
    uint32_t version{1};
    uint32_t max_fleets{MAX_FLEET_SIZE};
    uint32_t sync_interval_ms{TARGET_SYNC_INTERVAL_MS};
    uint32_t recovery_timeout_ms{5000};
    uint32_t monitoring_interval_ms{MONITORING_INTERVAL_MS};
    uint8_t power_save_mode{0};
    mesh_network_config_t mesh_config;
    struct {
        bool enable_performance_tracking{true};
        bool enable_power_monitoring{true};
        bool enable_auto_recovery{true};
    } monitor_config;
} __attribute__((packed, aligned(8)));

/**
 * @brief Thread-safe fleet management handle
 */
struct fleet_handle_t {
    char name[MAX_FLEET_NAME_LENGTH];
    uint32_t num_devices{0};
    mesh_fleet_t* mesh_fleet{nullptr};
    uint32_t sync_latency_ms{0};
    uint8_t state{0};
    uint8_t power_state{0};
    recovery_state_t recovery;
    monitor_stats_t stats;
    std::mutex fleet_mutex;
} __attribute__((aligned(8)));

/**
 * @brief Fleet options for creation and management
 */
struct fleet_options_t {
    uint32_t max_devices{MAX_FLEET_SIZE};
    uint32_t sync_interval_ms{TARGET_SYNC_INTERVAL_MS};
    bool enable_auto_recovery{true};
    bool enable_power_management{true};
    FleetPowerState initial_power_state{FleetPowerState::BALANCED};
};

/**
 * @brief Initialize the fleet coordination service
 * @param config Service configuration parameters
 * @return 0 on success, error code on failure
 */
__attribute__((init))
__attribute__((warn_unused_result))
int fleet_coordinator_init(fleet_coordinator_config_t* config);

/**
 * @brief Create a new fleet instance
 * @param fleet_name Unique fleet identifier
 * @param options Fleet configuration options
 * @return Fleet handle or nullptr on failure
 */
__attribute__((warn_unused_result))
fleet_handle_t* fleet_coordinator_create(const char* fleet_name, fleet_options_t* options);

/**
 * @brief Join an existing fleet
 * @param handle Fleet handle
 * @param device_id Device identifier
 * @return 0 on success, error code on failure
 */
__attribute__((warn_unused_result))
int fleet_coordinator_join(fleet_handle_t* handle, const char* device_id);

/**
 * @brief Leave current fleet
 * @param handle Fleet handle
 * @param device_id Device identifier
 * @return 0 on success, error code on failure
 */
int fleet_coordinator_leave(fleet_handle_t* handle, const char* device_id);

/**
 * @brief Update fleet power state
 * @param handle Fleet handle
 * @param state New power state
 * @return 0 on success, error code on failure
 */
int fleet_coordinator_set_power_state(fleet_handle_t* handle, FleetPowerState state);

/**
 * @brief Get current fleet statistics
 * @param handle Fleet handle
 * @return Const reference to monitoring statistics
 */
const monitor_stats_t& fleet_coordinator_get_stats(const fleet_handle_t* handle);

/**
 * @brief Error codes for fleet coordination
 */
enum FleetError {
    FLEET_SUCCESS = 0,
    FLEET_ERROR_INIT = -1,
    FLEET_ERROR_CONFIG = -2,
    FLEET_ERROR_MEMORY = -3,
    FLEET_ERROR_FULL = -4,
    FLEET_ERROR_NOT_FOUND = -5,
    FLEET_ERROR_NETWORK = -6,
    FLEET_ERROR_RECOVERY = -7,
    FLEET_ERROR_POWER = -8
};

/**
 * @brief Fleet states
 */
enum FleetState {
    FLEET_STATE_INIT = 0,
    FLEET_STATE_ACTIVE = 1,
    FLEET_STATE_DEGRADED = 2,
    FLEET_STATE_RECOVERY = 3,
    FLEET_STATE_ERROR = 4
};

} // namespace fleet
} // namespace tald

#endif // TALD_FLEET_COORDINATOR_H