/*
 * TALD UNIA - Kernel-Level Mesh Network Implementation
 * Version: 1.0
 * FreeBSD 9.0
 *
 * Provides high-performance P2P communication with 32-device fleet support
 * and sub-50ms latency guarantees through kernel-optimized mesh networking.
 */

#include <sys/types.h>      // FreeBSD 9.0
#include <sys/param.h>      // FreeBSD 9.0
#include <sys/kernel.h>     // FreeBSD 9.0
#include <sys/mutex.h>      // FreeBSD 9.0
#include <sys/malloc.h>     // FreeBSD 9.0
#include <sys/bus_dma.h>    // FreeBSD 9.0
#include "mesh_network.h"
#include "mesh_protocol.h"

/* Global state for the mesh network subsystem */
static struct {
    struct mtx global_lock;
    mesh_network_config_t config;
    mesh_fleet_t* fleets[MESH_MAX_FLEETS];
    uint32_t active_fleets;
    uma_zone_t packet_zone;
    bus_dma_tag_t dma_tag;
    struct callout timer;
    bool initialized;
} mesh_state;

/* Memory type for kernel allocations */
MALLOC_DEFINE(M_MESH, "mesh_network", "TALD UNIA Mesh Network");

/* Forward declarations of static functions */
static void mesh_timer_callback(void *arg);
static int mesh_setup_dma(void);
static int mesh_init_memory_pools(void);
static void mesh_cleanup_resources(void);

/**
 * Initialize the mesh network subsystem
 * @param config Subsystem configuration
 * @return 0 on success, error code on failure
 */
int
mesh_network_init(mesh_network_config_t* config)
{
    int error;

    /* Parameter validation */
    if (config == NULL || config->version != MESH_NETWORK_VERSION) {
        return MESH_ERROR_CONFIG;
    }

    /* Initialize global lock */
    mtx_init(&mesh_state.global_lock, "mesh_global", NULL, MTX_DEF);
    mtx_lock(&mesh_state.global_lock);

    /* Prevent double initialization */
    if (mesh_state.initialized) {
        mtx_unlock(&mesh_state.global_lock);
        return MESH_ERROR_INIT;
    }

    /* Copy configuration */
    bcopy(config, &mesh_state.config, sizeof(mesh_network_config_t));

    /* Initialize memory pools */
    error = mesh_init_memory_pools();
    if (error != 0) {
        goto cleanup;
    }

    /* Setup DMA resources */
    error = mesh_setup_dma();
    if (error != 0) {
        goto cleanup;
    }

    /* Initialize mesh protocol */
    error = mesh_protocol_init(&config->protocol_config);
    if (error != 0) {
        goto cleanup;
    }

    /* Initialize periodic timer */
    callout_init(&mesh_state.timer, CALLOUT_MPSAFE);
    callout_reset(&mesh_state.timer, hz/10, mesh_timer_callback, NULL);

    mesh_state.initialized = true;
    mtx_unlock(&mesh_state.global_lock);
    return 0;

cleanup:
    mesh_cleanup_resources();
    mtx_unlock(&mesh_state.global_lock);
    return error;
}

/**
 * Create a new mesh network fleet
 * @param fleet_id Unique fleet identifier
 * @param config Fleet configuration
 * @return Fleet handle or NULL on failure
 */
mesh_fleet_t*
mesh_network_create_fleet(const char* fleet_id, mesh_fleet_config_t* config)
{
    mesh_fleet_t* fleet;
    int error;

    if (!mesh_state.initialized || fleet_id == NULL || config == NULL) {
        return NULL;
    }

    mtx_lock(&mesh_state.global_lock);

    /* Check fleet limit */
    if (mesh_state.active_fleets >= MESH_MAX_FLEETS) {
        mtx_unlock(&mesh_state.global_lock);
        return NULL;
    }

    /* Allocate fleet structure */
    fleet = malloc(sizeof(mesh_fleet_t), M_MESH, M_WAITOK | M_ZERO);
    if (fleet == NULL) {
        mtx_unlock(&mesh_state.global_lock);
        return NULL;
    }

    /* Initialize fleet mutex */
    mtx_init(&fleet->fleet_mutex, "fleet_mutex", NULL, MTX_DEF);

    /* Setup fleet DMA buffer */
    error = bus_dmamem_alloc(mesh_state.dma_tag, &fleet->dma_base,
                            BUS_DMA_WAITOK | BUS_DMA_ZERO, NULL);
    if (error != 0) {
        free(fleet, M_MESH);
        mtx_unlock(&mesh_state.global_lock);
        return NULL;
    }

    /* Initialize fleet state */
    strlcpy(fleet->fleet_id, fleet_id, sizeof(fleet->fleet_id));
    fleet->num_devices = 0;
    fleet->state = MESH_FLEET_STATE_INIT;
    fleet->devices = malloc(sizeof(mesh_peer_t) * MESH_MAX_DEVICES_PER_FLEET,
                          M_MESH, M_WAITOK | M_ZERO);

    if (fleet->devices == NULL) {
        bus_dmamem_free(mesh_state.dma_tag, fleet->dma_base, NULL);
        free(fleet, M_MESH);
        mtx_unlock(&mesh_state.global_lock);
        return NULL;
    }

    /* Add to global fleet array */
    mesh_state.fleets[mesh_state.active_fleets++] = fleet;
    mtx_unlock(&mesh_state.global_lock);

    return fleet;
}

/**
 * Route data through the mesh network
 * @param fleet Fleet handle
 * @param data Data buffer
 * @param len Data length
 * @return Number of bytes routed or error code
 */
ssize_t
mesh_network_route(mesh_fleet_t* fleet, struct mbuf* data, size_t len)
{
    ssize_t bytes_sent = 0;
    int i;

    if (!mesh_state.initialized || fleet == NULL || data == NULL) {
        return MESH_ERROR_PARAM;
    }

    mtx_lock(&fleet->fleet_mutex);

    if (fleet->state != MESH_FLEET_STATE_ACTIVE) {
        mtx_unlock(&fleet->fleet_mutex);
        return MESH_ERROR_STATE;
    }

    /* Route to all fleet devices */
    for (i = 0; i < fleet->num_devices; i++) {
        mesh_peer_t* peer = &fleet->devices[i];
        ssize_t result;

        /* Skip inactive peers */
        if (peer->state != MESH_STATE_CONNECTED) {
            continue;
        }

        /* Send data with latency guarantee */
        result = mesh_protocol_send(peer, data, len);
        if (result > 0) {
            bytes_sent += result;
        }
    }

    mtx_unlock(&fleet->fleet_mutex);
    return bytes_sent;
}

/**
 * Timer callback for periodic maintenance
 */
static void
mesh_timer_callback(void *arg)
{
    int i;

    mtx_lock(&mesh_state.global_lock);

    /* Update fleet statistics */
    for (i = 0; i < mesh_state.active_fleets; i++) {
        mesh_fleet_t* fleet = mesh_state.fleets[i];
        uint32_t total_latency = 0;
        int active_devices = 0;
        int j;

        mtx_lock(&fleet->fleet_mutex);

        for (j = 0; j < fleet->num_devices; j++) {
            mesh_peer_t* peer = &fleet->devices[j];
            if (peer->state == MESH_STATE_CONNECTED) {
                total_latency += peer->latency_ms;
                active_devices++;
            }
        }

        /* Update average fleet latency */
        if (active_devices > 0) {
            fleet->avg_latency_ms = total_latency / active_devices;
        }

        /* Update fleet state based on health */
        if (active_devices == 0) {
            fleet->state = MESH_FLEET_STATE_ERROR;
        } else if (fleet->avg_latency_ms > MESH_TARGET_LATENCY_MS) {
            fleet->state = MESH_FLEET_STATE_DEGRADED;
        } else {
            fleet->state = MESH_FLEET_STATE_ACTIVE;
        }

        mtx_unlock(&fleet->fleet_mutex);
    }

    mtx_unlock(&mesh_state.global_lock);

    /* Reschedule timer */
    callout_reset(&mesh_state.timer, hz/10, mesh_timer_callback, NULL);
}

/**
 * Initialize DMA resources
 */
static int
mesh_setup_dma(void)
{
    return bus_dma_tag_create(NULL,                /* parent tag */
                             MESH_DMA_ALIGNMENT,   /* alignment */
                             0,                    /* boundary */
                             BUS_SPACE_MAXADDR,    /* lowaddr */
                             BUS_SPACE_MAXADDR,    /* highaddr */
                             NULL, NULL,           /* filter, filterarg */
                             MESH_MEMPOOL_SIZE,    /* maxsize */
                             1,                    /* nsegments */
                             MESH_MEMPOOL_SIZE,    /* maxsegsz */
                             0,                    /* flags */
                             NULL, NULL,           /* lockfunc, lockarg */
                             &mesh_state.dma_tag); /* tag */
}

/**
 * Initialize kernel memory pools
 */
static int
mesh_init_memory_pools(void)
{
    mesh_state.packet_zone = uma_zcreate("mesh_packets",
                                        MESH_MEMPOOL_SIZE,
                                        NULL, NULL, NULL, NULL,
                                        MESH_DMA_ALIGNMENT - 1,
                                        UMA_ZONE_MAXBYTES);
    return (mesh_state.packet_zone != NULL) ? 0 : MESH_ERROR_MEMORY;
}

/**
 * Cleanup allocated resources
 */
static void
mesh_cleanup_resources(void)
{
    int i;

    callout_drain(&mesh_state.timer);

    for (i = 0; i < mesh_state.active_fleets; i++) {
        mesh_fleet_t* fleet = mesh_state.fleets[i];
        if (fleet != NULL) {
            mtx_destroy(&fleet->fleet_mutex);
            if (fleet->devices != NULL) {
                free(fleet->devices, M_MESH);
            }
            if (fleet->dma_base != NULL) {
                bus_dmamem_free(mesh_state.dma_tag, fleet->dma_base, NULL);
            }
            free(fleet, M_MESH);
        }
    }

    if (mesh_state.packet_zone != NULL) {
        uma_zdestroy(mesh_state.packet_zone);
    }
    if (mesh_state.dma_tag != NULL) {
        bus_dma_tag_destroy(mesh_state.dma_tag);
    }

    mtx_destroy(&mesh_state.global_lock);
    mesh_state.initialized = false;
}