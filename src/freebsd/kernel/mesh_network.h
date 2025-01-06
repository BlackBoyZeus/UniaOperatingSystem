/*
 * TALD UNIA - Kernel-Level Mesh Network Subsystem
 * 
 * Provides high-performance P2P communication with 32-device fleet support
 * and sub-50ms latency guarantees. Implements core mesh network topology
 * management and routing logic with enhanced kernel safety mechanisms.
 *
 * Dependencies:
 * - FreeBSD 9.0: sys/types.h, sys/param.h, sys/kernel.h, sys/mutex.h, sys/mbuf.h
 * - Internal: mesh_protocol.h (v1), webrtc_native.h (M98)
 */

#ifndef _MESH_NETWORK_H_
#define _MESH_NETWORK_H_

#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/mutex.h>
#include <sys/mbuf.h>
#include "mesh_protocol.h"
#include "webrtc_native.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Version and System Constants */
#define MESH_NETWORK_VERSION          1
#define MESH_MAX_FLEETS              128
#define MESH_MAX_DEVICES_PER_FLEET   32
#define MESH_TARGET_LATENCY_MS       50
#define MESH_ROUTE_CACHE_SIZE        1024
#define MESH_KMEM_SIZE              (16 * 1024 * 1024)  /* 16MB */
#define MESH_MAX_INTERRUPTS         1024
#define MESH_PREEMPT_DISABLE        0x01

/* Forward declarations */
struct mesh_network_config;
struct mesh_fleet;
typedef struct mesh_network_config mesh_network_config_t;
typedef struct mesh_fleet mesh_fleet_t;

/**
 * Mesh Network Configuration Structure
 * Kernel-level configuration for the mesh network subsystem
 */
typedef struct mesh_network_config {
    uint32_t version;                  /* Subsystem version */
    uint32_t max_fleets;              /* Maximum number of fleets */
    uint32_t devices_per_fleet;       /* Maximum devices per fleet */
    uint32_t target_latency_ms;       /* Target P2P latency */
    mesh_protocol_config_t protocol_config;  /* Protocol configuration */
    webrtc_config_t webrtc_config;    /* WebRTC configuration */
    size_t kmem_size;                 /* Kernel memory pool size */
    uint32_t interrupt_priority;      /* Interrupt priority level */
    uint32_t dma_buffer_size;         /* DMA buffer size */
} __attribute__((packed)) mesh_network_config_t;

/**
 * Mesh Fleet Structure
 * Kernel-safe structure representing a mesh network fleet
 */
typedef struct mesh_fleet {
    char fleet_id[64];                /* Unique fleet identifier */
    uint32_t num_devices;             /* Current number of devices */
    mesh_peer_t* devices;             /* Array of fleet devices */
    uint32_t avg_latency_ms;          /* Average fleet latency */
    uint8_t state;                    /* Fleet state */
    struct mtx fleet_mutex;           /* Fleet mutex */
    struct malloc_type* kmem_type;    /* Kernel memory type */
    uint32_t interrupt_mask;          /* Interrupt mask */
    void* dma_base;                   /* DMA buffer base address */
} __attribute__((packed)) mesh_fleet_t;

/**
 * Initialize the mesh network subsystem
 * @param config Subsystem configuration
 * @param kmem_type Kernel memory type
 * @return 0 on success, error code on failure
 */
__init __must_check
int mesh_network_init(mesh_network_config_t* config, struct malloc_type* kmem_type);

/**
 * Create a new mesh network fleet
 * @param fleet_id Unique fleet identifier
 * @param config Fleet configuration
 * @param kmem_type Kernel memory type
 * @return Fleet handle or NULL on failure
 */
__must_check __locks_excluded(mesh_mutex)
mesh_fleet_t* mesh_network_create_fleet(
    const char* fleet_id,
    mesh_fleet_config_t* config,
    struct malloc_type* kmem_type
);

/**
 * Route data through the mesh network
 * @param fleet Fleet handle
 * @param data Data buffer
 * @param len Data length
 * @return Number of bytes routed or error code
 */
__must_check __interrupt_safe
ssize_t mesh_network_route(mesh_fleet_t* fleet, struct mbuf* data, size_t len);

/* Error codes */
#define MESH_SUCCESS                 0
#define MESH_ERROR_INIT            -1
#define MESH_ERROR_CONFIG          -2
#define MESH_ERROR_MEMORY          -3
#define MESH_ERROR_FLEET           -4
#define MESH_ERROR_ROUTE           -5
#define MESH_ERROR_INTERRUPT       -6
#define MESH_ERROR_DMA             -7

/* Fleet states */
#define MESH_FLEET_STATE_INIT       0x00
#define MESH_FLEET_STATE_ACTIVE     0x01
#define MESH_FLEET_STATE_DEGRADED   0x02
#define MESH_FLEET_STATE_ERROR      0x03

/* Kernel attributes */
#define __kernel_export           __attribute__((visibility("default")))
#define __kernel_packed          __attribute__((packed))
#define __interrupt_safe         __attribute__((interrupt))

/* Export symbols for kernel module use */
__kernel_export extern const mesh_network_config_t* mesh_get_default_config(void);
__kernel_export extern int mesh_get_fleet_stats(mesh_fleet_t* fleet);
__kernel_export extern uint32_t mesh_get_active_fleets(void);
__kernel_export extern int mesh_set_fleet_priority(mesh_fleet_t* fleet, uint8_t priority);

#ifdef __cplusplus
}
#endif

#endif /* _MESH_NETWORK_H_ */