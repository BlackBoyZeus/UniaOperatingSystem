/*
 * TALD UNIA - Mesh Network Protocol Implementation
 * 
 * Implements low-level protocol handling for mesh network topology with
 * 32-device fleet support and sub-50ms latency guarantees.
 *
 * Dependencies:
 * - FreeBSD 9.0: sys/types.h, sys/param.h, sys/socket.h
 * - WebRTC M98: webrtc_native.h
 */

#ifndef _MESH_PROTOCOL_H_
#define _MESH_PROTOCOL_H_

#include <sys/types.h>
#include <sys/param.h>
#include <sys/socket.h>
#include "webrtc_native.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Protocol Version and Constants */
#define MESH_PROTOCOL_VERSION        1
#define MESH_MAX_PEERS              32
#define MESH_MAX_LATENCY_MS         50
#define MESH_BUFFER_SIZE            65536
#define MESH_HEARTBEAT_INTERVAL_MS  100

/* Protocol States */
#define MESH_STATE_INIT             0x00
#define MESH_STATE_CONNECTING       0x01
#define MESH_STATE_CONNECTED        0x02
#define MESH_STATE_DISCONNECTED     0x03
#define MESH_STATE_ERROR            0x04

/* Protocol Flags */
#define MESH_FLAG_PRIORITY          0x01
#define MESH_FLAG_RELIABLE          0x02
#define MESH_FLAG_ENCRYPTED         0x04
#define MESH_FLAG_COMPRESSED        0x08

/**
 * Mesh Protocol Configuration
 * Configuration structure for the mesh protocol with kernel-safe parameters
 */
typedef struct mesh_protocol_config {
    uint32_t version;                  /* Protocol version */
    uint32_t max_peers;               /* Maximum number of peers */
    uint32_t max_latency_ms;          /* Maximum allowed latency */
    uint32_t buffer_size;             /* Protocol buffer size */
    uint32_t heartbeat_interval_ms;   /* Heartbeat interval */
    uint32_t retry_count;             /* Connection retry count */
    webrtc_config_t webrtc_config;    /* WebRTC configuration */
    uint32_t interrupt_priority;      /* Interrupt priority level */
    uint32_t memory_pool_size;        /* Memory pool size */
} __attribute__((packed)) mesh_protocol_config_t;

/**
 * Mesh Peer Structure
 * Structure representing a mesh protocol peer with enhanced state tracking
 */
typedef struct mesh_peer {
    char id[64];                      /* Peer identifier */
    webrtc_connection_t* connection;  /* WebRTC connection handle */
    webrtc_datachannel_t* data_channel; /* Data channel handle */
    uint32_t protocol_version;        /* Peer protocol version */
    uint32_t latency_ms;             /* Current latency */
    uint32_t last_heartbeat_ms;      /* Last heartbeat timestamp */
    uint32_t connection_quality;      /* Connection quality metric */
    uint32_t retry_count;            /* Connection retry counter */
    uint8_t state;                   /* Peer state */
    uint8_t priority;                /* Peer priority */
    uint8_t flags;                   /* Protocol flags */
} __attribute__((packed)) mesh_peer_t;

/**
 * Initialize the mesh protocol subsystem
 * @param config Protocol configuration
 * @return 0 on success, error code on failure
 */
__init __must_check
int mesh_protocol_init(mesh_protocol_config_t* config);

/**
 * Establish protocol-level connection with peer
 * @param peer_id Peer identifier
 * @param config Peer configuration
 * @return 0 on success, error code on failure
 */
__must_check __kernel
int mesh_protocol_connect(const char* peer_id, mesh_peer_config_t* config);

/**
 * Send data to peer with latency guarantees
 * @param peer Peer handle
 * @param data Data buffer
 * @param len Data length
 * @return Number of bytes sent or error code
 */
__must_check __kernel
ssize_t mesh_protocol_send(mesh_peer_t* peer, const void* data, size_t len);

/* Error Codes */
#define MESH_SUCCESS                 0
#define MESH_ERROR_INIT            -1
#define MESH_ERROR_CONFIG          -2
#define MESH_ERROR_MEMORY          -3
#define MESH_ERROR_PEER            -4
#define MESH_ERROR_TIMEOUT         -5
#define MESH_ERROR_LATENCY         -6
#define MESH_ERROR_VERSION         -7
#define MESH_ERROR_STATE           -8

/* Kernel Attributes */
#define __kernel_export           __attribute__((visibility("default")))
#define __kernel_packed          __attribute__((packed))

/* Exported Functions */
__kernel_export extern const mesh_protocol_config_t* mesh_get_default_config(void);
__kernel_export extern int mesh_get_peer_stats(mesh_peer_t* peer);
__kernel_export extern uint32_t mesh_get_active_peers(void);
__kernel_export extern int mesh_set_peer_priority(mesh_peer_t* peer, uint8_t priority);

#ifdef __cplusplus
}
#endif

#endif /* _MESH_PROTOCOL_H_ */