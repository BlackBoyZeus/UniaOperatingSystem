/*
 * TALD UNIA - Mesh Network Protocol Implementation
 * 
 * FreeBSD kernel driver implementing mesh networking protocol with 32-device fleet
 * support and sub-50ms latency guarantees. Provides interrupt-safe operations and
 * comprehensive latency monitoring.
 *
 * Dependencies:
 * - FreeBSD 9.0
 * - WebRTC M98
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/socket.h>
#include "mesh_protocol.h"
#include "webrtc_native.h"

/* Global state with cache line alignment for performance */
static volatile mesh_protocol_state_t g_mesh_protocol_state __aligned(CACHE_LINE_SIZE);
static mesh_peer_t g_peer_list[MESH_MAX_PEERS] __aligned(CACHE_LINE_SIZE);
static atomic_t g_peer_count;
static sx_t g_protocol_mutex;
static mesh_memory_pool_t g_memory_pool;
static mesh_latency_stats_t g_latency_stats;

/* Protocol version validation */
#define MESH_VERSION_CHECK(ver) ((ver) == MESH_PROTOCOL_VERSION)

/* Memory pool configuration */
#define MEMORY_POOL_SIZE (MESH_MAX_PEERS * MESH_BUFFER_SIZE * 2)
#define MEMORY_BLOCK_SIZE MESH_BUFFER_SIZE

__init __must_check
int mesh_protocol_init(mesh_protocol_config_t* config) {
    int error;

    /* Validate input parameters */
    if (!config || !MESH_VERSION_CHECK(config->version)) {
        return MESH_ERROR_CONFIG;
    }

    /* Initialize protocol mutex */
    sx_init(&g_protocol_mutex, "mesh_protocol");

    /* Initialize memory pool */
    error = mesh_memory_pool_init(&g_memory_pool, MEMORY_POOL_SIZE, MEMORY_BLOCK_SIZE);
    if (error) {
        sx_destroy(&g_protocol_mutex);
        return MESH_ERROR_MEMORY;
    }

    /* Initialize WebRTC subsystem */
    error = webrtc_init(&config->webrtc_config, &g_memory_pool.mem_config);
    if (error) {
        mesh_memory_pool_destroy(&g_memory_pool);
        sx_destroy(&g_protocol_mutex);
        return MESH_ERROR_INIT;
    }

    /* Initialize protocol state */
    atomic_set(&g_peer_count, 0);
    memset((void*)&g_mesh_protocol_state, 0, sizeof(mesh_protocol_state_t));
    g_mesh_protocol_state.state = MESH_STATE_INIT;
    g_mesh_protocol_state.max_peers = config->max_peers;
    g_mesh_protocol_state.max_latency_ms = config->max_latency_ms;

    /* Initialize latency monitoring */
    memset(&g_latency_stats, 0, sizeof(mesh_latency_stats_t));
    g_latency_stats.monitoring_interval_ms = config->heartbeat_interval_ms;

    return MESH_SUCCESS;
}

__must_check __locks_excluded(g_protocol_mutex)
int mesh_protocol_connect(const char* peer_id, mesh_peer_config_t* config) {
    int error;
    mesh_peer_t* peer = NULL;
    uint32_t peer_idx;

    /* Validate parameters */
    if (!peer_id || !config) {
        return MESH_ERROR_PARAM;
    }

    /* Check peer limit */
    if (atomic_read(&g_peer_count) >= MESH_MAX_PEERS) {
        return MESH_ERROR_PEER;
    }

    /* Acquire protocol mutex */
    sx_xlock(&g_protocol_mutex);

    /* Find available peer slot */
    for (peer_idx = 0; peer_idx < MESH_MAX_PEERS; peer_idx++) {
        if (g_peer_list[peer_idx].state == MESH_STATE_DISCONNECTED) {
            peer = &g_peer_list[peer_idx];
            break;
        }
    }

    if (!peer) {
        sx_xunlock(&g_protocol_mutex);
        return MESH_ERROR_PEER;
    }

    /* Initialize peer structure */
    memset(peer, 0, sizeof(mesh_peer_t));
    strlcpy(peer->id, peer_id, sizeof(peer->id));
    peer->protocol_version = config->protocol_version;
    peer->state = MESH_STATE_CONNECTING;
    peer->priority = config->priority;
    peer->flags = config->flags;

    /* Create WebRTC connection */
    peer->connection = webrtc_create_connection(peer_id, &config->webrtc_config, NULL);
    if (!peer->connection) {
        peer->state = MESH_STATE_DISCONNECTED;
        sx_xunlock(&g_protocol_mutex);
        return MESH_ERROR_INIT;
    }

    /* Create data channel */
    error = webrtc_create_datachannel(peer->connection, "mesh_data", &peer->data_channel);
    if (error) {
        webrtc_destroy_connection(peer->connection);
        peer->state = MESH_STATE_DISCONNECTED;
        sx_xunlock(&g_protocol_mutex);
        return error;
    }

    /* Update peer count */
    atomic_inc(&g_peer_count);
    
    /* Start latency monitoring */
    mesh_latency_monitor_start(peer);

    sx_xunlock(&g_protocol_mutex);
    return MESH_SUCCESS;
}

__must_check __locks_excluded(g_protocol_mutex)
ssize_t mesh_protocol_send(mesh_peer_t* peer, const void* data, size_t len) {
    void* send_buffer;
    ssize_t sent_bytes;
    mesh_protocol_header_t* header;

    /* Validate parameters */
    if (!peer || !data || !len || len > MESH_BUFFER_SIZE) {
        return MESH_ERROR_PARAM;
    }

    /* Check peer state */
    if (peer->state != MESH_STATE_CONNECTED) {
        return MESH_ERROR_STATE;
    }

    /* Acquire protocol mutex */
    sx_slock(&g_protocol_mutex);

    /* Allocate send buffer */
    send_buffer = mesh_memory_pool_alloc(&g_memory_pool);
    if (!send_buffer) {
        sx_sunlock(&g_protocol_mutex);
        return MESH_ERROR_MEMORY;
    }

    /* Prepare protocol header */
    header = (mesh_protocol_header_t*)send_buffer;
    header->version = MESH_PROTOCOL_VERSION;
    header->flags = peer->flags;
    header->sequence = atomic_inc_return(&peer->sequence);
    header->timestamp = mesh_get_timestamp_ms();
    header->data_length = len;

    /* Copy data after header */
    memcpy(send_buffer + sizeof(mesh_protocol_header_t), data, len);

    /* Send via WebRTC */
    sent_bytes = webrtc_send_data(peer->data_channel, send_buffer, 
                                 sizeof(mesh_protocol_header_t) + len);

    /* Update statistics */
    if (sent_bytes > 0) {
        atomic_inc(&peer->packets_sent);
        atomic_add(sent_bytes, &peer->bytes_sent);
        mesh_update_latency_stats(peer, header->timestamp);
    }

    /* Free send buffer */
    mesh_memory_pool_free(&g_memory_pool, send_buffer);

    sx_sunlock(&g_protocol_mutex);
    return sent_bytes;
}

__must_check __locks_excluded(g_protocol_mutex)
int mesh_protocol_disconnect(mesh_peer_t* peer) {
    int error;

    /* Validate parameters */
    if (!peer) {
        return MESH_ERROR_PARAM;
    }

    /* Acquire protocol mutex */
    sx_xlock(&g_protocol_mutex);

    /* Check if already disconnected */
    if (peer->state == MESH_STATE_DISCONNECTED) {
        sx_xunlock(&g_protocol_mutex);
        return MESH_SUCCESS;
    }

    /* Stop latency monitoring */
    mesh_latency_monitor_stop(peer);

    /* Close WebRTC connection */
    if (peer->data_channel) {
        webrtc_close_datachannel(peer->data_channel);
        peer->data_channel = NULL;
    }

    if (peer->connection) {
        error = webrtc_destroy_connection(peer->connection);
        peer->connection = NULL;
        if (error) {
            sx_xunlock(&g_protocol_mutex);
            return error;
        }
    }

    /* Update peer state */
    peer->state = MESH_STATE_DISCONNECTED;
    atomic_dec(&g_peer_count);

    sx_xunlock(&g_protocol_mutex);
    return MESH_SUCCESS;
}