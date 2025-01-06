/*
 * TALD UNIA LiDAR Hardware Driver Test Suite
 * Version: 1.0.0
 *
 * Comprehensive test suite for validating LiDAR hardware functionality including:
 * - 30Hz scanning frequency
 * - 0.01cm resolution
 * - 5-meter effective range
 * - ≤50ms latency requirements
 * - Thermal and power monitoring
 * - Safety features
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/module.h>
#include <atf-c.h>
#include "../../drivers/lidar/lidar_hw.h"
#include "../../drivers/lidar/lidar_calibration.h"

/* Test Constants */
#define TEST_SCAN_FREQUENCY     30      /* Required 30Hz scanning */
#define TEST_RESOLUTION_MM      0.1     /* 0.01cm resolution */
#define TEST_RANGE_MM          5000     /* 5-meter range */
#define TEST_BUFFER_SIZE       1048576  /* 1MB scan buffer */
#define TEST_TIMEOUT_MS        5000     /* 5-second timeout */
#define TEST_POWER_THRESHOLD_MW 2500    /* Power threshold */
#define TEST_THERMAL_LIMIT_CELSIUS 60   /* Thermal limit */
#define TEST_CALIBRATION_TARGETS {500, 1000, 2500, 5000} /* Test distances */

/* Test Utilities */
static struct lidar_hw_config test_config;
static struct calibration_params test_cal_params;
static uint8_t *scan_buffer = NULL;
static struct timespec start_time, end_time;

/* Setup function for hardware initialization test */
ATF_TC_HEAD(lidar_hw_init_test, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests LiDAR hardware initialization and safety features");
}

ATF_TC_WITH_CLEANUP(lidar_hw_init_test)
{
    /* Initialize test configuration */
    memset(&test_config, 0, sizeof(test_config));
    test_config.scan_frequency_hz = TEST_SCAN_FREQUENCY;
    test_config.resolution_mm = TEST_RESOLUTION_MM;
    test_config.range_mm = TEST_RANGE_MM;
    test_config.buffer_size = TEST_BUFFER_SIZE;
    test_config.power_limit_mw = TEST_POWER_THRESHOLD_MW;
    test_config.temp_threshold_c = TEST_THERMAL_LIMIT_CELSIUS;
    test_config.safety_features = LIDAR_SAFETY_TEMP_MON | 
                                 LIDAR_SAFETY_POWER_MON | 
                                 LIDAR_SAFETY_INTERLOCKS;

    /* Test hardware initialization */
    int result = lidar_hw_init(&test_config, NULL);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Hardware initialization failed");

    /* Verify power monitoring */
    uint32_t power_reading;
    result = lidar_hw_get_power_reading(&power_reading);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Power reading failed");
    ATF_REQUIRE_MSG(power_reading < TEST_POWER_THRESHOLD_MW, 
                    "Power consumption exceeds threshold");

    /* Verify thermal monitoring */
    int32_t temperature;
    result = lidar_hw_get_temperature(&temperature);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Temperature reading failed");
    ATF_REQUIRE_MSG(temperature < TEST_THERMAL_LIMIT_CELSIUS, 
                    "Temperature exceeds threshold");

    /* Verify safety interlocks */
    uint32_t safety_status;
    result = lidar_hw_get_safety_status(&safety_status);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Safety status check failed");
    ATF_REQUIRE_MSG(safety_status & LIDAR_SAFETY_INTERLOCKS, 
                    "Safety interlocks not active");
}

ATF_TC_CLEANUP(lidar_hw_init_test)
{
    lidar_hw_shutdown();
}

/* Setup function for scanning test */
ATF_TC_HEAD(lidar_hw_scan_test, tc)
{
    atf_tc_set_md_var(tc, "descr", "Tests LiDAR scanning operations and performance");
}

ATF_TC_WITH_CLEANUP(lidar_hw_scan_test)
{
    /* Initialize calibration parameters */
    memset(&test_cal_params, 0, sizeof(test_cal_params));
    test_cal_params.version = 1;
    test_cal_params.distance_offset_mm = 0.0f;
    test_cal_params.angle_offset_deg = 0.0f;
    test_cal_params.temperature_coefficient = 1.0f;

    /* Initialize hardware with calibration */
    int result = lidar_hw_init(&test_config, &test_cal_params);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Hardware initialization failed");

    /* Allocate scan buffer */
    scan_buffer = malloc(TEST_BUFFER_SIZE);
    ATF_REQUIRE_MSG(scan_buffer != NULL, "Scan buffer allocation failed");

    /* Start scanning operation */
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    result = lidar_hw_start_scan(scan_buffer, TEST_BUFFER_SIZE);
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Scan start failed");

    /* Test scanning frequency */
    uint32_t scan_count = 0;
    struct timespec scan_start, scan_end;
    clock_gettime(CLOCK_MONOTONIC, &scan_start);

    for (int i = 0; i < TEST_SCAN_FREQUENCY; i++) {
        result = lidar_hw_wait_scan_complete(TEST_TIMEOUT_MS);
        ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Scan completion wait failed");
        scan_count++;
    }

    clock_gettime(CLOCK_MONOTONIC, &scan_end);
    double elapsed = (scan_end.tv_sec - scan_start.tv_sec) + 
                    (scan_end.tv_nsec - scan_start.tv_nsec) / 1e9;
    
    /* Verify scanning frequency */
    double actual_frequency = scan_count / elapsed;
    ATF_REQUIRE_MSG(fabs(actual_frequency - TEST_SCAN_FREQUENCY) < 0.1, 
                    "Scanning frequency outside tolerance");

    /* Test scan latency */
    struct timespec latency_start, latency_end;
    clock_gettime(CLOCK_MONOTONIC, &latency_start);
    result = lidar_hw_get_latest_scan(scan_buffer, TEST_BUFFER_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &latency_end);
    
    double latency_ms = ((latency_end.tv_sec - latency_start.tv_sec) * 1000.0) +
                       ((latency_end.tv_nsec - latency_start.tv_nsec) / 1e6);
    ATF_REQUIRE_MSG(latency_ms <= 50.0, "Scan latency exceeds 50ms requirement");

    /* Verify resolution and range */
    const uint32_t test_distances[] = TEST_CALIBRATION_TARGETS;
    for (size_t i = 0; i < sizeof(test_distances)/sizeof(test_distances[0]); i++) {
        result = lidar_hw_verify_measurement(test_distances[i], TEST_RESOLUTION_MM);
        ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, 
                       "Resolution verification failed at %u mm", test_distances[i]);
    }

    /* Stop scanning */
    result = lidar_hw_stop_scan();
    ATF_REQUIRE_MSG(result == LIDAR_ERR_SUCCESS, "Scan stop failed");
}

ATF_TC_CLEANUP(lidar_hw_scan_test)
{
    if (scan_buffer != NULL) {
        free(scan_buffer);
    }
    lidar_hw_shutdown();
}

/* Test suite initialization */
ATF_TP_ADD_TCS(tp)
{
    ATF_TP_ADD_TC(tp, lidar_hw_init_test);
    ATF_TP_ADD_TC(tp, lidar_hw_scan_test);
    return atf_no_error();
}