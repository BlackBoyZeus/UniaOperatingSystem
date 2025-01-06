/*
 * TALD UNIA - Mesh Protocol Test Suite
 * 
 * Comprehensive test suite for kernel-level mesh networking protocol
 * validating P2P communication, fleet management, and latency requirements.
 *
 * Dependencies:
 * - FreeBSD 9.0: sys/types.h, sys/param.h, sys/kernel.h
 * - Kyua 0.13: atf-c.h
 * - Internal: mesh_protocol.h, webrtc_native.h
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <atf-c.h>
#include "drivers/network/mesh_protocol.h"
#include "drivers/network/webrtc_native.h"

/* Test Constants */
#define TEST_PEER_COUNT              32
#define TEST_BUFFER_SIZE            65536
#define TEST_LATENCY_THRESHOLD_MS   50
#define TEST_MESSAGE_SIZE           1024
#define TEST_LATENCY_SAMPLES       1000
#define TEST_INTERRUPT_PRIORITY     5

/* Test Context Structure */
typedef struct test_peer_context {
    mesh_protocol_config_t config;
    mesh_peer_t* peers;
    uint32_t peer_count;
    uint8_t* test_buffer;
    struct timespec* latency_samples;
    uint32_t interrupt_count;
} test_peer_context_t;

/* Static Test Buffer */
static uint8_t __kernel_aligned(4096) test_data[TEST_BUFFER_SIZE];

/* Test Context Management */
static test_peer_context_t*
test_context_create(uint32_t num_peers)
{
    test_peer_context_t* ctx = malloc(sizeof(test_peer_context_t));
    ATF_REQUIRE(ctx != NULL);

    /* Initialize configuration */
    ctx->config.version = MESH_PROTOCOL_VERSION;
    ctx->config.max_peers = num_peers;
    ctx->config.max_latency_ms = TEST_LATENCY_THRESHOLD_MS;
    ctx->config.buffer_size = TEST_BUFFER_SIZE;
    ctx->config.interrupt_priority = TEST_INTERRUPT_PRIORITY;
    ctx->config.memory_pool_size = TEST_BUFFER_SIZE * 2;

    /* Allocate peer array */
    ctx->peers = malloc(sizeof(mesh_peer_t) * num_peers);
    ATF_REQUIRE(ctx->peers != NULL);
    ctx->peer_count = num_peers;

    /* Allocate test buffer */
    ctx->test_buffer = malloc(TEST_BUFFER_SIZE);
    ATF_REQUIRE(ctx->test_buffer != NULL);

    /* Allocate latency samples array */
    ctx->latency_samples = malloc(sizeof(struct timespec) * TEST_LATENCY_SAMPLES);
    ATF_REQUIRE(ctx->latency_samples != NULL);

    return ctx;
}

static void
test_context_destroy(test_peer_context_t* ctx)
{
    if (ctx) {
        free(ctx->peers);
        free(ctx->test_buffer);
        free(ctx->latency_samples);
        free(ctx);
    }
}

/* Test Cases */
ATF_TC_WITH_CLEANUP(test_mesh_protocol_init);
ATF_TC_HEAD(test_mesh_protocol_init, tc)
{
    atf_tc_set_md_var(tc, "descr", "Test mesh protocol initialization");
}

ATF_TC_BODY(test_mesh_protocol_init, tc)
{
    test_peer_context_t* ctx = test_context_create(TEST_PEER_COUNT);
    
    /* Test protocol initialization */
    int ret = mesh_protocol_init(&ctx->config);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);

    /* Verify configuration */
    ATF_REQUIRE_EQ(ctx->config.version, MESH_PROTOCOL_VERSION);
    ATF_REQUIRE_EQ(ctx->config.max_peers, TEST_PEER_COUNT);
    ATF_REQUIRE_EQ(ctx->config.max_latency_ms, TEST_LATENCY_THRESHOLD_MS);

    /* Test invalid configurations */
    mesh_protocol_config_t invalid_config = ctx->config;
    invalid_config.version = 0;
    ret = mesh_protocol_init(&invalid_config);
    ATF_REQUIRE_EQ(ret, MESH_ERROR_VERSION);

    invalid_config = ctx->config;
    invalid_config.max_peers = MESH_MAX_PEERS + 1;
    ret = mesh_protocol_init(&invalid_config);
    ATF_REQUIRE_EQ(ret, MESH_ERROR_CONFIG);

    test_context_destroy(ctx);
}

ATF_TC_CLEANUP(test_mesh_protocol_init, tc)
{
    /* Cleanup is handled by test_context_destroy */
}

ATF_TC_WITH_CLEANUP(test_mesh_protocol_connect);
ATF_TC_HEAD(test_mesh_protocol_connect, tc)
{
    atf_tc_set_md_var(tc, "descr", "Test peer connection establishment");
}

ATF_TC_BODY(test_mesh_protocol_connect, tc)
{
    test_peer_context_t* ctx = test_context_create(TEST_PEER_COUNT);
    int ret = mesh_protocol_init(&ctx->config);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);

    /* Test peer connection */
    for (uint32_t i = 0; i < ctx->peer_count; i++) {
        char peer_id[64];
        snprintf(peer_id, sizeof(peer_id), "test_peer_%u", i);
        ret = mesh_protocol_connect(peer_id, NULL);
        ATF_REQUIRE_EQ(ret, MESH_SUCCESS);
    }

    /* Verify connection states */
    uint32_t active_peers = mesh_get_active_peers();
    ATF_REQUIRE_EQ(active_peers, ctx->peer_count);

    test_context_destroy(ctx);
}

ATF_TC_CLEANUP(test_mesh_protocol_connect, tc)
{
    /* Cleanup is handled by test_context_destroy */
}

ATF_TC_WITH_CLEANUP(test_mesh_protocol_latency);
ATF_TC_HEAD(test_mesh_protocol_latency, tc)
{
    atf_tc_set_md_var(tc, "descr", "Test mesh protocol latency requirements");
}

ATF_TC_BODY(test_mesh_protocol_latency, tc)
{
    test_peer_context_t* ctx = test_context_create(2); // Test with 2 peers
    int ret = mesh_protocol_init(&ctx->config);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);

    /* Connect test peers */
    ret = mesh_protocol_connect("latency_peer_1", NULL);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);
    ret = mesh_protocol_connect("latency_peer_2", NULL);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);

    /* Perform latency measurements */
    struct timespec start, end;
    uint64_t total_latency = 0;
    
    for (uint32_t i = 0; i < TEST_LATENCY_SAMPLES; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        ret = mesh_protocol_send(&ctx->peers[0], test_data, TEST_MESSAGE_SIZE);
        ATF_REQUIRE(ret >= 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        uint64_t latency = (end.tv_sec - start.tv_sec) * 1000000000ULL +
                          (end.tv_nsec - start.tv_nsec);
        total_latency += latency;
        
        /* Store sample */
        ctx->latency_samples[i] = end;
        
        /* Verify against threshold */
        ATF_REQUIRE_LE(latency / 1000000ULL, TEST_LATENCY_THRESHOLD_MS);
    }

    /* Calculate average latency */
    uint64_t avg_latency = total_latency / TEST_LATENCY_SAMPLES / 1000000ULL;
    ATF_REQUIRE_LE(avg_latency, TEST_LATENCY_THRESHOLD_MS);

    test_context_destroy(ctx);
}

ATF_TC_CLEANUP(test_mesh_protocol_latency, tc)
{
    /* Cleanup is handled by test_context_destroy */
}

ATF_TC_WITH_CLEANUP(test_mesh_protocol_fleet);
ATF_TC_HEAD(test_mesh_protocol_fleet, tc)
{
    atf_tc_set_md_var(tc, "descr", "Test fleet management capabilities");
}

ATF_TC_BODY(test_mesh_protocol_fleet, tc)
{
    test_peer_context_t* ctx = test_context_create(TEST_PEER_COUNT);
    int ret = mesh_protocol_init(&ctx->config);
    ATF_REQUIRE_EQ(ret, MESH_SUCCESS);

    /* Create maximum size fleet */
    for (uint32_t i = 0; i < TEST_PEER_COUNT; i++) {
        char peer_id[64];
        snprintf(peer_id, sizeof(peer_id), "fleet_peer_%u", i);
        ret = mesh_protocol_connect(peer_id, NULL);
        ATF_REQUIRE_EQ(ret, MESH_SUCCESS);
        
        /* Verify peer state */
        ATF_REQUIRE_EQ(ctx->peers[i].state, MESH_STATE_CONNECTED);
    }

    /* Verify fleet size */
    uint32_t active_peers = mesh_get_active_peers();
    ATF_REQUIRE_EQ(active_peers, TEST_PEER_COUNT);

    /* Test fleet-wide message broadcast */
    for (uint32_t i = 0; i < TEST_PEER_COUNT; i++) {
        ret = mesh_protocol_send(&ctx->peers[i], test_data, TEST_MESSAGE_SIZE);
        ATF_REQUIRE(ret >= 0);
    }

    test_context_destroy(ctx);
}

ATF_TC_CLEANUP(test_mesh_protocol_fleet, tc)
{
    /* Cleanup is handled by test_context_destroy */
}

/* Test Suite Definition */
ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, test_mesh_protocol_init);
    ATF_TP_ADD_TC(tp, test_mesh_protocol_connect);
    ATF_TP_ADD_TC(tp, test_mesh_protocol_latency);
    ATF_TP_ADD_TC(tp, test_mesh_protocol_fleet);

    return atf_no_error();
}