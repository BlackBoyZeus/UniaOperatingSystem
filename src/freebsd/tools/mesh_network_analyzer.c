/*
 * TALD UNIA - Mesh Network Performance Analyzer
 * Version: 1.0.0
 * 
 * Advanced performance analysis and diagnostic tool for mesh networking system
 * with support for 32-device fleets and sub-50ms latency monitoring.
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/socket.h>
#include "mesh_protocol.h"
#include "webrtc_native.h"

/* Version and Constants */
#define ANALYZER_VERSION              "1.0.0"
#define ANALYZER_SAMPLE_INTERVAL_MS   100
#define ANALYZER_HISTORY_SIZE         1000
#define ANALYZER_MAX_PEERS           32
#define ANALYZER_LATENCY_THRESHOLD_MS 50
#define ANALYZER_ALERT_INTERVAL_MS    1000
#define ANALYZER_CRDT_SYNC_TIMEOUT_MS 200

/* Private structures for internal metrics tracking */
typedef struct {
    struct timeval timestamp;
    uint32_t peer_count;
    uint32_t latencies[ANALYZER_MAX_PEERS];
    uint64_t bytes_sent[ANALYZER_MAX_PEERS];
    uint64_t bytes_received[ANALYZER_MAX_PEERS];
    float packet_loss[ANALYZER_MAX_PEERS];
    uint32_t crdt_sync_times[ANALYZER_MAX_PEERS];
} __attribute__((packed)) analyzer_metrics_snapshot_t;

typedef struct {
    analyzer_metrics_snapshot_t history[ANALYZER_HISTORY_SIZE];
    uint32_t history_index;
    uint32_t total_samples;
    struct timeval last_alert;
    uint32_t alert_count;
} __attribute__((packed)) analyzer_history_t;

/* Static analyzer state */
static analyzer_config_t current_config;
static analyzer_history_t metrics_history;
static uint8_t analyzer_initialized = 0;

/* Initialize the mesh network analyzer */
__init
int analyzer_init(analyzer_config_t* config) {
    if (!config) {
        return -EINVAL;
    }

    /* Validate configuration parameters */
    if (config->sample_interval_ms < 10 || 
        config->history_size > ANALYZER_HISTORY_SIZE ||
        config->alert_threshold_ms > 1000) {
        return -EINVAL;
    }

    /* Initialize mesh protocol monitoring */
    mesh_protocol_config_t mesh_config = {
        .version = MESH_PROTOCOL_VERSION,
        .max_peers = ANALYZER_MAX_PEERS,
        .max_latency_ms = ANALYZER_LATENCY_THRESHOLD_MS,
        .buffer_size = MESH_BUFFER_SIZE,
        .heartbeat_interval_ms = config->sample_interval_ms
    };

    int ret = mesh_protocol_init(&mesh_config);
    if (ret != 0) {
        return ret;
    }

    /* Initialize metrics history */
    memset(&metrics_history, 0, sizeof(analyzer_history_t));
    memcpy(&current_config, config, sizeof(analyzer_config_t));
    
    analyzer_initialized = 1;
    return 0;
}

/* Hot path for metrics collection */
__hot
int analyzer_collect_metrics(mesh_metrics_t* metrics) {
    if (!analyzer_initialized || !metrics) {
        return -EINVAL;
    }

    struct timeval now;
    gettimeofday(&now, NULL);

    /* Create new metrics snapshot */
    analyzer_metrics_snapshot_t snapshot;
    memset(&snapshot, 0, sizeof(analyzer_metrics_snapshot_t));
    snapshot.timestamp = now;
    
    /* Collect per-peer metrics */
    uint32_t total_latency = 0;
    uint32_t max_latency = 0;
    uint64_t total_bytes = 0;
    float total_loss = 0;
    uint32_t active_peers = 0;

    for (uint32_t i = 0; i < ANALYZER_MAX_PEERS; i++) {
        mesh_peer_t* peer = mesh_protocol_get_peer(i);
        if (!peer || peer->state != MESH_STATE_CONNECTED) {
            continue;
        }

        /* Get WebRTC stats */
        webrtc_connection_t* conn = peer->connection;
        webrtc_stats_t stats;
        if (webrtc_get_connection_stats(conn) == 0) {
            snapshot.latencies[i] = stats.current_latency_ms;
            snapshot.bytes_sent[i] = stats.bytes_sent;
            snapshot.bytes_received[i] = stats.bytes_received;
            snapshot.packet_loss[i] = stats.packet_loss_percentage;
            
            total_latency += stats.current_latency_ms;
            max_latency = MAX(max_latency, stats.current_latency_ms);
            total_bytes += (stats.bytes_sent + stats.bytes_received);
            total_loss += stats.packet_loss_percentage;
            active_peers++;
        }
    }

    /* Update metrics history */
    metrics_history.history[metrics_history.history_index] = snapshot;
    metrics_history.history_index = (metrics_history.history_index + 1) % ANALYZER_HISTORY_SIZE;
    metrics_history.total_samples++;

    /* Update output metrics */
    metrics->peer_count = active_peers;
    metrics->avg_latency_ms = active_peers ? (total_latency / active_peers) : 0;
    metrics->max_latency_ms = max_latency;
    metrics->total_bytes_sent = total_bytes;
    metrics->packet_loss_rate = active_peers ? (total_loss / active_peers) : 0;
    
    /* Check for alerts */
    if (max_latency > current_config.alert_threshold_ms) {
        struct timeval diff;
        timersub(&now, &metrics_history.last_alert, &diff);
        if (diff.tv_sec * 1000 + diff.tv_usec / 1000 >= ANALYZER_ALERT_INTERVAL_MS) {
            metrics_history.last_alert = now;
            metrics_history.alert_count++;
        }
    }

    return 0;
}

/* Generate detailed network analysis report */
int analyzer_generate_report(const char* output_path) {
    if (!analyzer_initialized || !output_path) {
        return -EINVAL;
    }

    FILE* report = fopen(output_path, "w");
    if (!report) {
        return -EIO;
    }

    /* Report header */
    fprintf(report, "TALD UNIA Mesh Network Analysis Report\n");
    fprintf(report, "Version: %s\n", ANALYZER_VERSION);
    fprintf(report, "Timestamp: %ld\n\n", time(NULL));

    /* Fleet statistics */
    uint32_t active_peers = 0;
    float avg_fleet_latency = 0;
    float avg_packet_loss = 0;
    uint64_t total_throughput = 0;

    /* Calculate fleet-wide statistics */
    for (uint32_t i = 0; i < metrics_history.total_samples; i++) {
        analyzer_metrics_snapshot_t* snapshot = &metrics_history.history[i];
        active_peers = MAX(active_peers, snapshot->peer_count);
        
        for (uint32_t p = 0; p < ANALYZER_MAX_PEERS; p++) {
            avg_fleet_latency += snapshot->latencies[p];
            avg_packet_loss += snapshot->packet_loss[p];
            total_throughput += (snapshot->bytes_sent[p] + snapshot->bytes_received[p]);
        }
    }

    if (metrics_history.total_samples > 0) {
        avg_fleet_latency /= (metrics_history.total_samples * ANALYZER_MAX_PEERS);
        avg_packet_loss /= (metrics_history.total_samples * ANALYZER_MAX_PEERS);
    }

    /* Write summary statistics */
    fprintf(report, "Fleet Summary:\n");
    fprintf(report, "- Active Peers: %u\n", active_peers);
    fprintf(report, "- Average Fleet Latency: %.2f ms\n", avg_fleet_latency);
    fprintf(report, "- Average Packet Loss: %.2f%%\n", avg_packet_loss);
    fprintf(report, "- Total Throughput: %lu bytes\n", total_throughput);
    fprintf(report, "- Alert Count: %u\n\n", metrics_history.alert_count);

    /* Generate per-peer analysis */
    fprintf(report, "Per-Peer Analysis:\n");
    for (uint32_t i = 0; i < ANALYZER_MAX_PEERS; i++) {
        mesh_peer_t* peer = mesh_protocol_get_peer(i);
        if (!peer || peer->state != MESH_STATE_CONNECTED) {
            continue;
        }

        fprintf(report, "\nPeer %u:\n", i);
        fprintf(report, "- ID: %s\n", peer->id);
        fprintf(report, "- Connection Quality: %u%%\n", peer->connection_quality);
        fprintf(report, "- Average Latency: %.2f ms\n", 
                metrics_history.history[metrics_history.history_index].latencies[i]);
        fprintf(report, "- Packet Loss: %.2f%%\n",
                metrics_history.history[metrics_history.history_index].packet_loss[i]);
    }

    fclose(report);
    return 0;
}