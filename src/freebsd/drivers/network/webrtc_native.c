/*
 * TALD UNIA - Native WebRTC Implementation for FreeBSD Kernel
 * Version: M98 (libwebrtc)
 *
 * Kernel-space WebRTC implementation optimized for low-latency P2P mesh networking
 * with support for 32-device fleets and deterministic performance characteristics.
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/malloc.h>
#include <sys/interrupt.h>
#include <libwebrtc/webrtc.h>
#include "webrtc_native.h"

/* Memory allocation type for WebRTC subsystem */
MALLOC_DEFINE(M_WEBRTC, "webrtc", "WebRTC kernel subsystem");

/* Static globals for connection management */
static struct {
    webrtc_connection_impl *connections[WEBRTC_MAX_CONNECTIONS];
    uint32_t active_connections;
    struct mtx connection_lock;
    webrtc_mempool_t *mempool;
    bool initialized;
} webrtc_state = { 0 };

/* Performance monitoring structure */
typedef struct performance_metrics {
    uint64_t latency_samples[1024];
    uint32_t sample_index;
    uint64_t total_packets;
    uint64_t dropped_packets;
    struct timespec last_update;
} performance_metrics_t;

/* Implementation of webrtc_connection */
struct webrtc_connection_impl {
    webrtc_connection_t public_conn;
    void *peer_connection;
    webrtc_datachannel_t *data_channels;
    uint32_t channel_count;
    uint8_t state;
    webrtc_mempool_t *mempool;
    interrupt_context_t *int_ctx;
    performance_metrics_t *metrics;
    struct callout watchdog_timer;
    uint32_t packet_queue[WEBRTC_MAX_PENDING_PACKETS];
    uint32_t queue_head;
    uint32_t queue_tail;
};

/* Interrupt handler for WebRTC events */
static void
webrtc_interrupt_handler(void *arg)
{
    struct webrtc_connection_impl *conn = (struct webrtc_connection_impl *)arg;
    critical_enter();
    
    /* Process pending packets with latency constraints */
    while (conn->queue_head != conn->queue_tail) {
        uint32_t packet = conn->packet_queue[conn->queue_head];
        if (webrtc_process_packet(conn->peer_connection, packet) == 0) {
            conn->queue_head = (conn->queue_head + 1) % WEBRTC_MAX_PENDING_PACKETS;
        }
    }
    
    critical_exit();
}

/* Initialize WebRTC subsystem */
__init __must_check
int 
webrtc_init(webrtc_config_t *config, webrtc_mempool_t *mempool)
{
    if (webrtc_state.initialized) {
        return WEBRTC_ERROR_INIT;
    }

    /* Validate WebRTC version */
    if (strcmp(WEBRTC_VERSION_CHECK, webrtc_get_version()) != 0) {
        return WEBRTC_ERROR_VERSION;
    }

    /* Initialize connection lock */
    mtx_init(&webrtc_state.connection_lock, "webrtc_lock", NULL, MTX_DEF);

    /* Configure kernel memory pool */
    webrtc_state.mempool = mempool;
    if (kernel_memory_init(mempool, WEBRTC_KERNEL_HEAP_SIZE) != 0) {
        return WEBRTC_ERROR_MEMORY;
    }

    /* Initialize WebRTC core with kernel optimizations */
    struct webrtc_kernel_config kern_config = {
        .max_connections = WEBRTC_MAX_CONNECTIONS,
        .interrupt_priority = WEBRTC_INTERRUPT_PRIORITY,
        .zero_copy = true,
        .kernel_threading = true
    };

    if (webrtc_core_init(&kern_config) != 0) {
        kernel_memory_cleanup(mempool);
        return WEBRTC_ERROR_INIT;
    }

    webrtc_state.initialized = true;
    return WEBRTC_SUCCESS;
}

/* Create new WebRTC peer connection */
__must_check __interrupt_safe
webrtc_connection_t*
webrtc_create_connection(const char *peer_id, webrtc_connection_config_t *config, int interrupt_priority)
{
    struct webrtc_connection_impl *conn;
    
    /* Check fleet size limit */
    mtx_lock(&webrtc_state.connection_lock);
    if (webrtc_state.active_connections >= WEBRTC_MAX_CONNECTIONS) {
        mtx_unlock(&webrtc_state.connection_lock);
        return NULL;
    }

    /* Allocate connection structure from pre-initialized pool */
    conn = kernel_memory_alloc(webrtc_state.mempool, sizeof(*conn));
    if (conn == NULL) {
        mtx_unlock(&webrtc_state.connection_lock);
        return NULL;
    }

    /* Initialize connection structure */
    bzero(conn, sizeof(*conn));
    strlcpy(conn->public_conn.peer_id, peer_id, sizeof(conn->public_conn.peer_id));
    conn->mempool = webrtc_state.mempool;
    
    /* Configure interrupt handling */
    conn->int_ctx = interrupt_context_create(interrupt_priority);
    if (conn->int_ctx == NULL) {
        kernel_memory_free(webrtc_state.mempool, conn);
        mtx_unlock(&webrtc_state.connection_lock);
        return NULL;
    }

    /* Initialize performance metrics */
    conn->metrics = kernel_memory_alloc(webrtc_state.mempool, sizeof(performance_metrics_t));
    if (conn->metrics == NULL) {
        interrupt_context_destroy(conn->int_ctx);
        kernel_memory_free(webrtc_state.mempool, conn);
        mtx_unlock(&webrtc_state.connection_lock);
        return NULL;
    }

    /* Configure WebRTC peer connection */
    struct webrtc_peer_config peer_config = {
        .ice_servers = WEBRTC_ICE_SERVERS,
        .max_latency_ms = WEBRTC_MAX_LATENCY_MS,
        .buffer_size = WEBRTC_BUFFER_SIZE,
        .zero_copy = true
    };

    conn->peer_connection = webrtc_peer_create(&peer_config);
    if (conn->peer_connection == NULL) {
        kernel_memory_free(webrtc_state.mempool, conn->metrics);
        interrupt_context_destroy(conn->int_ctx);
        kernel_memory_free(webrtc_state.mempool, conn);
        mtx_unlock(&webrtc_state.connection_lock);
        return NULL;
    }

    /* Initialize watchdog timer */
    callout_init_mtx(&conn->watchdog_timer, &webrtc_state.connection_lock, 0);
    
    /* Add to active connections */
    webrtc_state.connections[webrtc_state.active_connections++] = conn;
    mtx_unlock(&webrtc_state.connection_lock);

    return &conn->public_conn;
}

/* Connection implementation cleanup */
static void
webrtc_connection_cleanup(struct webrtc_connection_impl *conn)
{
    callout_drain(&conn->watchdog_timer);
    webrtc_peer_destroy(conn->peer_connection);
    interrupt_context_destroy(conn->int_ctx);
    kernel_memory_free(conn->mempool, conn->metrics);
    kernel_memory_free(conn->mempool, conn);
}

/* Module initialization and cleanup */
static int
webrtc_native_init(void)
{
    return 0;
}

static void
webrtc_native_cleanup(void)
{
    mtx_lock(&webrtc_state.connection_lock);
    for (uint32_t i = 0; i < webrtc_state.active_connections; i++) {
        webrtc_connection_cleanup(webrtc_state.connections[i]);
    }
    webrtc_state.active_connections = 0;
    mtx_unlock(&webrtc_state.connection_lock);
    mtx_destroy(&webrtc_state.connection_lock);
}

SYSINIT(webrtc_native_init, SI_SUB_DRIVERS, SI_ORDER_MIDDLE, webrtc_native_init, NULL);
SYSUNINIT(webrtc_native_cleanup, SI_SUB_DRIVERS, SI_ORDER_MIDDLE, webrtc_native_cleanup, NULL);

MODULE_VERSION(webrtc_native, 1);