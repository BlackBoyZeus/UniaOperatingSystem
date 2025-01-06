/*
 * TALD UNIA Core Kernel Test Suite
 * Version: 1.0.0
 *
 * Comprehensive test suite for the TALD UNIA core kernel module with
 * enhanced interrupt safety, thermal monitoring, and resource validation.
 */

#include <sys/types.h>
#include <sys/module.h>
#include <sys/systm.h>
#include <kern/test.h>
#include <sys/smp.h>
#include "tald_core.h"

/* Test configuration constants */
#define TEST_TALD_CORE_VERSION "1.0.0"
#define TEST_MEMORY_LIMIT 4294967296UL  /* 4GB */
#define TEST_THERMAL_LIMIT 85           /* 85°C */
#define TEST_INTERRUPT_PRIORITY 5

/* Global test state */
static struct tald_core_config test_config;
static int test_interrupt_context;

/* Test setup helper functions */
static void setup_test_config(void) {
    bzero(&test_config, sizeof(test_config));
    test_config.version = TEST_TALD_CORE_VERSION;
    test_config.memory_limit = TEST_MEMORY_LIMIT;
    test_config.power_state = POWER_BALANCED;
    test_config.thermal_config.gpu_temp_limit_c = TEST_THERMAL_LIMIT;
    test_config.thermal_config.lidar_temp_limit_c = TEST_THERMAL_LIMIT;
    test_config.thermal_config.soc_temp_limit_c = TEST_THERMAL_LIMIT;
    test_config.safety_config.thermal_threshold_c = TEST_THERMAL_LIMIT;
    test_config.safety_config.safety_flags = TALD_SAFETY_THERMAL_MON | 
                                           TALD_SAFETY_POWER_MON | 
                                           TALD_SAFETY_WATCHDOG;
    test_config.watchdog.timeout_ms = TALD_WATCHDOG_TIMEOUT;
    test_config.watchdog.enabled = true;
}

static void cleanup_test_state(void) {
    tald_core_shutdown();
    bzero(&test_config, sizeof(test_config));
    test_interrupt_context = 0;
}

/*
 * Test successful initialization of TALD core system
 */
TEST_CASE(test_tald_core_init_success) {
    int result;
    critical_enter();  /* Enter critical section for interrupt safety */
    
    /* Setup test configuration */
    setup_test_config();
    
    /* Verify kernel context */
    TEST_ASSERT(curthread->td_critnest > 0, "Not in critical section");
    
    /* Test initialization */
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_SUCCESS, "Core initialization failed");
    
    /* Verify system state */
    TEST_ASSERT(test_config.power_state == POWER_BALANCED, 
                "Incorrect power state");
    TEST_ASSERT(test_config.watchdog.enabled, "Watchdog not enabled");
    TEST_ASSERT(test_config.safety_config.safety_flags & TALD_SAFETY_THERMAL_MON,
                "Thermal monitoring not enabled");
    
    cleanup_test_state();
    critical_exit();
}

/*
 * Test core initialization with invalid configuration
 */
TEST_CASE(test_tald_core_init_invalid_config) {
    int result;
    critical_enter();
    
    /* Test null configuration */
    result = tald_core_init(NULL);
    TEST_ASSERT(result == TALD_ERROR_CONFIG, "Null config not detected");
    
    /* Test invalid version */
    setup_test_config();
    test_config.version = "0.0.0";
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_ERROR_CONFIG, "Invalid version not detected");
    
    /* Test invalid memory limit */
    setup_test_config();
    test_config.memory_limit = 0;
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_ERROR_MEMORY, "Invalid memory limit not detected");
    
    /* Test invalid thermal configuration */
    setup_test_config();
    test_config.thermal_config.gpu_temp_limit_c = 0;
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_ERROR_THERMAL, "Invalid thermal limit not detected");
    
    cleanup_test_state();
    critical_exit();
}

/*
 * Test power state transitions with thermal monitoring
 */
TEST_CASE(test_tald_core_power_states) {
    int result;
    critical_enter();
    
    /* Initialize with test configuration */
    setup_test_config();
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_SUCCESS, "Core initialization failed");
    
    /* Test transition to performance mode */
    test_config.power_state = POWER_PERFORMANCE;
    result = tald_set_power_state(POWER_PERFORMANCE);
    TEST_ASSERT(result == TALD_SUCCESS, "Performance mode transition failed");
    TEST_ASSERT(test_config.power_state == POWER_PERFORMANCE, 
                "Incorrect power state after transition");
    
    /* Test thermal throttling */
    test_config.thermal_config.gpu_temp_limit_c = TEST_THERMAL_LIMIT - 1;
    result = tald_set_power_state(POWER_PERFORMANCE);
    TEST_ASSERT(result == TALD_ERROR_THERMAL, 
                "Thermal throttling not triggered");
    TEST_ASSERT(test_config.power_state == POWER_BALANCED, 
                "Thermal throttling did not adjust power state");
    
    cleanup_test_state();
    critical_exit();
}

/*
 * Test memory limit enforcement with barrier testing
 */
TEST_CASE(test_tald_core_memory_limits) {
    int result;
    critical_enter();
    
    /* Initialize with test configuration */
    setup_test_config();
    result = tald_core_init(&test_config);
    TEST_ASSERT(result == TALD_SUCCESS, "Core initialization failed");
    
    /* Test memory allocation at limit */
    size_t test_size = TEST_MEMORY_LIMIT;
    result = tald_allocate_memory(test_size);
    TEST_ASSERT(result == TALD_ERROR_MEMORY, 
                "Memory limit not enforced");
    
    /* Test fragmentation handling */
    test_size = TEST_MEMORY_LIMIT / 2;
    result = tald_allocate_memory(test_size);
    TEST_ASSERT(result == TALD_SUCCESS, 
                "Valid memory allocation failed");
    
    /* Verify memory barrier effectiveness */
    test_size = TEST_MEMORY_LIMIT;
    result = tald_allocate_memory(test_size);
    TEST_ASSERT(result == TALD_ERROR_MEMORY, 
                "Memory barrier violation not detected");
    
    cleanup_test_state();
    critical_exit();
}

/* Test case registration */
TEST_SET(tald_core_tests) {
    TEST_ADD(test_tald_core_init_success);
    TEST_ADD(test_tald_core_init_invalid_config);
    TEST_ADD(test_tald_core_power_states);
    TEST_ADD(test_tald_core_memory_limits);
}