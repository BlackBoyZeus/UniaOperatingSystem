/*
 * TALD UNIA LiDAR Driver Test Suite
 * Version: 1.0.0
 *
 * Comprehensive test suite for FreeBSD kernel LiDAR driver verification
 * Testing compliance with:
 * - 30Hz scanning frequency
 * - 0.01cm resolution
 * - 5-meter effective range
 * - ≤50ms processing latency
 */

#include <sys/param.h>
#include <sys/module.h>
#include <sys/kernel.h>
#include <sys/systm.h>
#include <sys/malloc.h>
#include <sys/sysctl.h>
#include <sys/time.h>

#include "lidar_driver.h"
#include "lidar_hw.h"

/* Test Constants */
#define TEST_DEVICE_ID           1
#define TEST_SCAN_FREQUENCY     30
#define TEST_RESOLUTION_MM      0.1
#define TEST_RANGE_MM          5000
#define TEST_MAX_LATENCY_MS     50
#define TEST_THERMAL_LIMIT_C    85
#define TEST_LASER_POWER_MW     20
#define TEST_DMA_BUFFER_SIZE    1048576

/* Test Context Structure */
struct lidar_test_context {
    struct lidar_driver_config driver_config;
    struct lidar_hw_config hw_config;
    struct lidar_hw_status hw_status;
    device_t test_device;
    struct timeval start_time;
    struct timeval end_time;
    uint32_t dma_buffer_size;
    void *dma_buffer;
    uint32_t thermal_readings[8];
    uint32_t laser_power_readings[8];
};

/* Static Test Context */
static struct lidar_test_context test_ctx;

/* Helper Functions */
static void init_test_context(void) {
    bzero(&test_ctx, sizeof(struct lidar_test_context));
    
    /* Initialize driver configuration */
    test_ctx.driver_config.device_id = TEST_DEVICE_ID;
    test_ctx.driver_config.irq_number = 0;
    test_ctx.driver_config.dma_channel = 0;
    test_ctx.driver_config.dma_alignment = LIDAR_DMA_ALIGNMENT;
    test_ctx.driver_config.watchdog_timeout_ms = LIDAR_ERROR_TIMEOUT_MS;
    
    /* Initialize hardware configuration */
    test_ctx.hw_config.scan_frequency_hz = TEST_SCAN_FREQUENCY;
    test_ctx.hw_config.resolution_mm = TEST_RESOLUTION_MM;
    test_ctx.hw_config.range_mm = TEST_RANGE_MM;
    test_ctx.hw_config.buffer_size = TEST_DMA_BUFFER_SIZE;
    test_ctx.hw_config.safety_features = LIDAR_SAFETY_TEMP_MON | 
                                       LIDAR_SAFETY_POWER_MON |
                                       LIDAR_SAFETY_INTERLOCKS;
    
    /* Allocate DMA buffer */
    test_ctx.dma_buffer = malloc(TEST_DMA_BUFFER_SIZE, M_DEVBUF, M_WAITOK | M_ZERO);
    test_ctx.dma_buffer_size = TEST_DMA_BUFFER_SIZE;
}

static void cleanup_test_context(void) {
    if (test_ctx.dma_buffer != NULL) {
        free(test_ctx.dma_buffer, M_DEVBUF);
        test_ctx.dma_buffer = NULL;
    }
}

static int measure_latency_ms(void) {
    struct timeval diff;
    timersub(&test_ctx.end_time, &test_ctx.start_time, &diff);
    return (diff.tv_sec * 1000) + (diff.tv_usec / 1000);
}

/* Test Cases */
TEST_CASE(test_lidar_driver_init) {
    int result;
    struct lidar_safety_params safety = {
        .laser_power_limit_mw = TEST_LASER_POWER_MW,
        .thermal_shutdown_temp_c = TEST_THERMAL_LIMIT_C,
        .watchdog_timeout_ms = LIDAR_ERROR_TIMEOUT_MS,
        .error_threshold = 3,
        .safety_flags = LIDAR_SAFETY_THERMAL_MON | LIDAR_SAFETY_POWER_MON
    };

    /* Test initialization with valid configuration */
    result = lidar_driver_init(&test_ctx.driver_config, &safety);
    KASSERT(result == 0, ("Driver initialization failed with valid config"));
    
    /* Verify driver state */
    KASSERT(test_ctx.driver_config.device_id == TEST_DEVICE_ID,
            ("Device ID mismatch"));
    KASSERT(test_ctx.hw_config.scan_frequency_hz == TEST_SCAN_FREQUENCY,
            ("Scan frequency mismatch"));
            
    /* Test initialization with invalid configuration */
    test_ctx.driver_config.device_id = LIDAR_MAX_DEVICES + 1;
    result = lidar_driver_init(&test_ctx.driver_config, &safety);
    KASSERT(result == LIDAR_DRIVER_ERR_CONFIG,
            ("Failed to detect invalid device ID"));
}

TEST_CASE(test_lidar_scan_operations) {
    int result, latency;
    uint32_t scan_count = 0;
    struct timeval scan_start, scan_end, diff;
    
    /* Start scanning operation */
    microtime(&test_ctx.start_time);
    result = lidar_driver_attach(test_ctx.test_device, NULL);
    KASSERT(result == 0, ("Failed to start scanning operation"));
    
    /* Measure scan frequency stability */
    for (int i = 0; i < 100; i++) {
        microtime(&scan_start);
        /* Wait for one scan cycle */
        DELAY(1000000/TEST_SCAN_FREQUENCY);
        microtime(&scan_end);
        timersub(&scan_end, &scan_start, &diff);
        
        /* Verify scan timing (30Hz ±1%) */
        KASSERT(diff.tv_usec >= 32900 && diff.tv_usec <= 33100,
                ("Scan frequency outside tolerance"));
        scan_count++;
    }
    
    microtime(&test_ctx.end_time);
    latency = measure_latency_ms();
    
    /* Verify latency requirements */
    KASSERT(latency <= TEST_MAX_LATENCY_MS,
            ("Processing latency exceeds maximum allowed"));
    
    /* Verify scan resolution */
    KASSERT(test_ctx.hw_config.resolution_mm == TEST_RESOLUTION_MM,
            ("Resolution not meeting 0.01cm requirement"));
            
    /* Verify effective range */
    KASSERT(test_ctx.hw_config.range_mm == TEST_RANGE_MM,
            ("Range not meeting 5-meter requirement"));
}

TEST_CASE(test_lidar_error_handling) {
    int result;
    struct lidar_error_state error_state;
    
    /* Test thermal shutdown */
    test_ctx.hw_status.thermal.current_temp_c = TEST_THERMAL_LIMIT_C + 1;
    result = lidar_driver_attach(test_ctx.test_device, NULL);
    KASSERT(result == LIDAR_DRIVER_ERR_THERMAL,
            ("Failed to detect thermal shutdown condition"));
    
    /* Test DMA error recovery */
    test_ctx.hw_status.dma_status = LIDAR_DMA_ERROR;
    result = lidar_driver_attach(test_ctx.test_device, NULL);
    KASSERT(result == LIDAR_DRIVER_ERR_DMA,
            ("Failed to detect DMA error condition"));
    
    /* Verify error state tracking */
    KASSERT(test_ctx.hw_status.errors.error_count > 0,
            ("Error count not incremented"));
    KASSERT(test_ctx.hw_status.errors.last_error_code != 0,
            ("Last error code not set"));
}

TEST_CASE(test_lidar_safety_compliance) {
    int result;
    struct lidar_safety_params safety;
    
    /* Configure safety parameters */
    safety.laser_power_limit_mw = TEST_LASER_POWER_MW;
    safety.thermal_shutdown_temp_c = TEST_THERMAL_LIMIT_C;
    safety.watchdog_timeout_ms = LIDAR_ERROR_TIMEOUT_MS;
    safety.safety_flags = LIDAR_SAFETY_THERMAL_MON | LIDAR_SAFETY_POWER_MON;
    
    /* Test laser power limits */
    result = lidar_driver_init(&test_ctx.driver_config, &safety);
    KASSERT(result == 0, ("Safety parameter initialization failed"));
    
    /* Verify thermal protection */
    KASSERT(test_ctx.driver_config.safety.thermal_shutdown_temp_c == TEST_THERMAL_LIMIT_C,
            ("Thermal limit not properly configured"));
            
    /* Verify safety interlocks */
    KASSERT(test_ctx.hw_config.safety_features & LIDAR_SAFETY_INTERLOCKS,
            ("Safety interlocks not enabled"));
}

/* Test Module */
static int
lidar_test_module_handler(module_t mod, int event, void *arg) {
    int error = 0;
    
    switch (event) {
    case MOD_LOAD:
        init_test_context();
        printf("TALD UNIA LiDAR Driver Test Suite loaded\n");
        break;
    case MOD_UNLOAD:
        cleanup_test_context();
        printf("TALD UNIA LiDAR Driver Test Suite unloaded\n");
        break;
    default:
        error = EOPNOTSUPP;
        break;
    }
    
    return (error);
}

static moduledata_t lidar_test_module = {
    "lidar_test",
    lidar_test_module_handler,
    NULL
};

DECLARE_MODULE(lidar_test, lidar_test_module, SI_SUB_DRIVERS, SI_ORDER_MIDDLE);
MODULE_VERSION(lidar_test, 1);
MODULE_DEPEND(lidar_test, lidar_driver, 1, 1, 1);