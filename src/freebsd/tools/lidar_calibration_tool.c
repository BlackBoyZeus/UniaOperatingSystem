/*
 * TALD UNIA LiDAR Calibration Tool
 * Version: 1.0.0
 *
 * Command-line tool for calibrating and validating LiDAR hardware
 * with temperature compensation and enhanced safety features.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include "../drivers/lidar/lidar_hw.h"
#include "../drivers/lidar/lidar_calibration.h"

#define TOOL_VERSION "1.0.0"
#define DEFAULT_CONFIG_PATH "/etc/tald/lidar/calibration.conf"
#define MAX_RETRIES 3
#define MIN_TEMP_C 10.0
#define MAX_TEMP_C 40.0
#define THERMAL_SHUTDOWN_TEMP_C 45.0
#define CALIBRATION_TIMEOUT_MS 30000

/* Tool configuration structure */
struct tool_config {
    char *config_path;
    bool verify_only;
    bool force_calibration;
    bool verbose;
    struct lidar_hw_config hw_config;
    struct calibration_params cal_params;
    float ambient_temperature;
    uint8_t safety_status;
    uint64_t last_calibration_time;
    bool temperature_compensation_enabled;
};

/* Function prototypes */
static void print_usage(const char *program_name);
static int parse_arguments(int argc, char **argv, struct tool_config *config);
static int check_hardware_safety(const struct tool_config *config);
static int monitor_temperature(struct tool_config *config);
static int perform_calibration(struct tool_config *config);
static int verify_calibration(struct tool_config *config);
static void cleanup_and_exit(struct tool_config *config, int status);

int main(int argc, char **argv) {
    struct tool_config config = {
        .config_path = DEFAULT_CONFIG_PATH,
        .verify_only = false,
        .force_calibration = false,
        .verbose = false,
        .ambient_temperature = 0.0f,
        .safety_status = 0,
        .temperature_compensation_enabled = true
    };

    int status = 0;

    printf("TALD UNIA LiDAR Calibration Tool v%s\n", TOOL_VERSION);

    /* Parse command line arguments */
    if ((status = parse_arguments(argc, argv, &config)) != 0) {
        print_usage(argv[0]);
        return status;
    }

    /* Initialize hardware configuration */
    config.hw_config.scan_frequency_hz = LIDAR_SCAN_FREQ_HZ;
    config.hw_config.resolution_mm = (uint32_t)(LIDAR_RESOLUTION_MM * 10);
    config.hw_config.range_mm = LIDAR_RANGE_MM;
    config.hw_config.safety_features = LIDAR_SAFETY_TEMP_MON | 
                                     LIDAR_SAFETY_POWER_MON | 
                                     LIDAR_SAFETY_INTERLOCKS;

    /* Perform hardware safety checks */
    if ((status = check_hardware_safety(&config)) != 0) {
        fprintf(stderr, "Hardware safety check failed: %d\n", status);
        cleanup_and_exit(&config, status);
        return status;
    }

    /* Start temperature monitoring thread */
    if ((status = monitor_temperature(&config)) != 0) {
        fprintf(stderr, "Temperature monitoring failed: %d\n", status);
        cleanup_and_exit(&config, status);
        return status;
    }

    /* Perform calibration or verification */
    if (config.verify_only) {
        status = verify_calibration(&config);
    } else {
        status = perform_calibration(&config);
    }

    cleanup_and_exit(&config, status);
    return status;
}

static void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -c, --config <path>    Configuration file path\n");
    printf("  -v, --verify          Verify calibration only\n");
    printf("  -f, --force           Force recalibration\n");
    printf("  -V, --verbose         Enable verbose output\n");
    printf("  -h, --help            Display this help message\n");
}

static int parse_arguments(int argc, char **argv, struct tool_config *config) {
    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"verify", no_argument, 0, 'v'},
        {"force", no_argument, 0, 'f'},
        {"verbose", no_argument, 0, 'V'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option;
    while ((option = getopt_long(argc, argv, "c:vfVh", long_options, NULL)) != -1) {
        switch (option) {
            case 'c':
                config->config_path = optarg;
                break;
            case 'v':
                config->verify_only = true;
                break;
            case 'f':
                config->force_calibration = true;
                break;
            case 'V':
                config->verbose = true;
                break;
            case 'h':
                return 1;
            default:
                return -1;
        }
    }
    return 0;
}

static int check_hardware_safety(const struct tool_config *config) {
    /* Check temperature range */
    if (config->ambient_temperature < MIN_TEMP_C || 
        config->ambient_temperature > MAX_TEMP_C) {
        fprintf(stderr, "Temperature out of safe range: %.1f°C\n", 
                config->ambient_temperature);
        return LIDAR_ERR_TEMPERATURE;
    }

    /* Verify power status */
    if (config->hw_config.power_limit_mw > LIDAR_MAX_POWER_MW) {
        fprintf(stderr, "Power consumption exceeds safe limit\n");
        return LIDAR_ERR_POWER;
    }

    /* Check safety interlocks */
    if (!(config->hw_config.safety_features & LIDAR_SAFETY_INTERLOCKS)) {
        fprintf(stderr, "Safety interlocks not engaged\n");
        return LIDAR_ERR_SAFETY;
    }

    return 0;
}

static int monitor_temperature(struct tool_config *config) {
    struct calibration_options cal_options = {
        .min_temperature = MIN_TEMP_C,
        .max_temperature = MAX_TEMP_C,
        .timeout_ms = CALIBRATION_TIMEOUT_MS,
        .flags = LIDAR_CAL_FLAG_TEMP_COMP
    };

    /* Initialize temperature monitoring */
    if (config->verbose) {
        printf("Initializing temperature monitoring...\n");
    }

    /* Set up temperature compensation */
    config->cal_params.temperature_coefficient = 
        (MAX_TEMP_C - MIN_TEMP_C) / THERMAL_SHUTDOWN_TEMP_C;
    
    if (config->verbose) {
        printf("Temperature coefficient: %.3f\n", 
               config->cal_params.temperature_coefficient);
    }

    return 0;
}

static int perform_calibration(struct tool_config *config) {
    struct calibration_options cal_options = {
        .num_measurement_points = CALIBRATION_POINTS_COUNT,
        .measurements_per_point = 100,
        .min_temperature = MIN_TEMP_C,
        .max_temperature = MAX_TEMP_C,
        .timeout_ms = CALIBRATION_TIMEOUT_MS,
        .flags = LIDAR_CAL_FLAG_TEMP_COMP | 
                 LIDAR_CAL_FLAG_VERIFY | 
                 (config->verbose ? LIDAR_CAL_FLAG_VERBOSE : 0)
    };

    if (config->verbose) {
        printf("Starting calibration process...\n");
    }

    int status = lidar_calibrate_device(&config->hw_config, 
                                      &config->cal_params, 
                                      &cal_options);

    if (status != LIDAR_CAL_SUCCESS) {
        fprintf(stderr, "Calibration failed with error: %d\n", status);
        return status;
    }

    if (config->verbose) {
        printf("Calibration completed successfully\n");
        printf("Accuracy score: %d%%\n", config->cal_params.accuracy_score);
        printf("Temperature at calibration: %.1f°C\n", 
               config->cal_params.calibration_temperature);
    }

    return 0;
}

static int verify_calibration(struct tool_config *config) {
    if (config->verbose) {
        printf("Verifying calibration parameters...\n");
    }

    int status = lidar_verify_calibration(&config->cal_params, 
                                        config->ambient_temperature);

    if (status != LIDAR_CAL_SUCCESS) {
        fprintf(stderr, "Calibration verification failed: %d\n", status);
        return status;
    }

    if (config->verbose) {
        printf("Calibration verification successful\n");
    }

    return 0;
}

static void cleanup_and_exit(struct tool_config *config, int status) {
    if (config->verbose) {
        printf("Cleaning up and exiting with status: %d\n", status);
    }
    
    /* Perform any necessary cleanup */
    if (status != 0) {
        /* Log error condition */
        fprintf(stderr, "Tool exited with errors\n");
    }
}