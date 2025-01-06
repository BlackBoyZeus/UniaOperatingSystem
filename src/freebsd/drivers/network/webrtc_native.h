/*
 * TALD UNIA - Native WebRTC Implementation for FreeBSD Kernel
 * 
 * This header provides kernel-space WebRTC functionality optimized for 
 * low-latency P2P mesh networking with support for 32-device fleets.
 *
 * External Dependencies:
 * - FreeBSD 9.0: sys/types.h, sys/param.h, sys/kernel.h
 * - WebRTC M98: libwebrtc
 */

#ifndef _WEBRTC_NATIVE_H_
#define _WEBRTC_NATIVE_H_

#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <libwebrtc/webrtc.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Global Constants */
#define WEBRTC_MAX_CONNECTIONS      32    /* Maximum concurrent peer connections */
#define WEBRTC_MAX_DATA_CHANNELS    4     /* Data channels per connection */
#define WEBRTC_MAX_LATENCY_MS      50    /* Maximum allowed P2P latency */
#define WEBRTC_BUFFER_SIZE         65536  /* Buffer size for data channels */
#define WEBRTC_ICE_SERVERS         "stun:stun.tald.unia:3478"
#define WEBRTC_VERSION_CHECK       "M98"  /* Required WebRTC version */
#define WEBRTC_KERNEL_HEAP_SIZE    8388608 /* 8MB kernel heap allocation */
#define WEBRTC_MAX_PENDING_PACKETS 1024   /* Maximum queued packets */
#define WEBRTC_INTERRUPT_PRIORITY  4      /* Interrupt priority level */

/* Forward declarations */
struct kernel_memory_config;
struct kernel_interrupt_context;
typedef struct kernel_memory_config kernel_memory_config_t;
typedef struct kernel_interrupt_context kernel_interrupt_context_t;

/**
 * WebRTC Configuration Structure
 * Enhanced configuration for kernel-space WebRTC operations
 */
typedef struct webrtc_config {
    uint32_t max_connections;        /* Maximum allowed connections */
    uint32_t max_data_channels;      /* Maximum data channels per connection */
    uint32_t max_latency_ms;         /* Maximum allowed latency */
    uint32_t buffer_size;            /* Buffer size for data transfer */
    char *ice_servers;               /* ICE server configuration */
    uint32_t kernel_heap_size;       /* Kernel heap allocation size */
    uint8_t interrupt_priority;      /* Interrupt priority level */
    uint32_t max_pending_packets;    /* Maximum pending packets in queue */
    bool enable_kernel_threading;    /* Enable kernel threading support */
    uint32_t connection_timeout_ms;  /* Connection timeout in milliseconds */
} webrtc_config_t;

/**
 * WebRTC Connection Structure
 * Kernel-optimized peer connection management
 */
typedef struct webrtc_connection {
    char peer_id[64];               /* Unique peer identifier */
    void *peer_connection;          /* WebRTC peer connection handle */
    void *data_channels;            /* Array of data channel handles */
    uint32_t latency_ms;            /* Current connection latency */
    uint8_t state;                  /* Connection state */
    void *mem_block;                /* Kernel memory allocation */
    void *int_handler;              /* Interrupt handler */
    uint32_t packet_queue_size;     /* Current packet queue size */
    uint64_t last_activity_ticks;   /* Last activity timestamp */
    struct timespec connection_start_time; /* Connection start time */
} webrtc_connection_t;

/**
 * Initialize WebRTC subsystem
 * @param config WebRTC configuration structure
 * @param mem_config Kernel memory configuration
 * @return 0 on success, error code on failure
 */
__init __must_check
int webrtc_init(webrtc_config_t *config, kernel_memory_config_t *mem_config);

/**
 * Create new WebRTC peer connection
 * @param peer_id Unique peer identifier
 * @param config Connection configuration
 * @param int_ctx Kernel interrupt context
 * @return Connection handle or NULL on failure
 */
__must_check __kernel_context
webrtc_connection_t* webrtc_create_connection(
    const char *peer_id,
    webrtc_config_t *config,
    kernel_interrupt_context_t *int_ctx
);

/* Connection state definitions */
enum webrtc_connection_state {
    WEBRTC_STATE_NEW = 0,
    WEBRTC_STATE_CONNECTING = 1,
    WEBRTC_STATE_CONNECTED = 2,
    WEBRTC_STATE_DISCONNECTED = 3,
    WEBRTC_STATE_FAILED = 4,
    WEBRTC_STATE_CLOSED = 5
};

/* Error codes */
#define WEBRTC_SUCCESS             0
#define WEBRTC_ERROR_INIT         -1
#define WEBRTC_ERROR_MEMORY       -2
#define WEBRTC_ERROR_PARAM        -3
#define WEBRTC_ERROR_STATE        -4
#define WEBRTC_ERROR_TIMEOUT      -5
#define WEBRTC_ERROR_INTERRUPT    -6
#define WEBRTC_ERROR_VERSION      -7

/* Kernel-specific attributes */
#define __kernel_export           __attribute__((visibility("default")))
#define __kernel_packed          __attribute__((packed))
#define __kernel_aligned(x)      __attribute__((aligned(x)))

/* Export symbols for kernel module use */
__kernel_export extern const webrtc_config_t* webrtc_get_default_config(void);
__kernel_export extern int webrtc_get_connection_stats(webrtc_connection_t *conn);
__kernel_export extern uint32_t webrtc_get_active_connections(void);

#ifdef __cplusplus
}
#endif

#endif /* _WEBRTC_NATIVE_H_ */