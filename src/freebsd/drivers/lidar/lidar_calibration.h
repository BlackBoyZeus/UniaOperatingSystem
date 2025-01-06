/*
 * TALD UNIA LiDAR Calibration System
 * Version: 1.0.0
 *
 * Calibration header file defining structures and functions for the LiDAR
 * hardware subsystem calibration with:
 * - Temperature-compensated calibration
 * - 0.01cm resolution accuracy
 * - 5-meter range validation
 * - 30Hz operation verification
 * - Enhanced safety features
 */

#ifndef _LIDAR_CALIBRATION_H_
#define _LIDAR_CALIBRATION_H_

#include <sys/types.h>
#include <sys/param.h>
#include "lidar_hw.h"

/* Version and Magic Numbers */
#define LIDAR_CALIBRATION_VERSION "1.0.0"
#define CALIBRATION_POINTS_COUNT  1024
#define MAX_CALIBRATION_RETRIES   3
#define CALIBRATION_TIMEOUT_MS    5000
#define MIN_CALIBRATION_TEMP_C    -10.0
#define MAX_CALIBRATION_TEMP_C    50.0
#define MIN_ACCURACY_SCORE        95
#define CALIBRATION_DATA_MAGIC    0x4C494441 /* 'LIDA' */

/* Temperature Compensation Structure */
struct temp_compensation {
    float base_temperature;           /* Base calibration temperature */
    float distance_coeff[3];          /* Distance correction coefficients */
    float angle_coeff[3];            /* Angle correction coefficients */
    float intensity_coeff[2];        /* Intensity correction coefficients */
} __packed;

/* Calibration Point Structure */
struct calibration_point {
    float measured_distance_mm;      /* Measured distance in mm */
    float reference_distance_mm;     /* Reference distance in mm */
    float measured_angle_deg;        /* Measured angle in degrees */
    float reference_angle_deg;       /* Reference angle in degrees */
    uint32_t intensity_value;        /* Measured intensity value */
    float temperature;               /* Temperature at measurement */
    uint32_t measurement_count;      /* Number of measurements taken */
    float standard_deviation;        /* Measurement standard deviation */
} __packed;

/* Calibration Parameters Structure */
struct calibration_params {
    uint32_t version;               /* Calibration data version */
    uint32_t magic;                 /* Magic number for validation */
    float distance_offset_mm;       /* Global distance offset */
    float angle_offset_deg;         /* Global angle offset */
    float intensity_scale;          /* Intensity scaling factor */
    float temperature_coefficient;  /* Global temperature coefficient */
    float calibration_temperature; /* Temperature at calibration */
    uint64_t calibration_timestamp; /* Unix timestamp of calibration */
    uint32_t accuracy_score;       /* Calibration accuracy score */
    uint32_t checksum;             /* CRC32 of calibration data */
    struct temp_compensation temp_comp; /* Temperature compensation data */
} __packed;

/* Calibration Options Structure */
struct calibration_options {
    uint32_t num_measurement_points;  /* Number of calibration points */
    uint32_t measurements_per_point;  /* Measurements per point */
    float min_temperature;           /* Minimum valid temperature */
    float max_temperature;           /* Maximum valid temperature */
    uint32_t timeout_ms;             /* Calibration timeout in ms */
    uint32_t flags;                  /* Calibration control flags */
} __packed;

/* Function Prototypes */

/*
 * Performs full calibration of the LiDAR device with temperature compensation
 * @param config Pointer to hardware configuration
 * @param params Pointer to calibration parameters
 * @param options Pointer to calibration options
 * @return 0 on success, negative error code on failure
 */
int lidar_calibrate_device(struct lidar_hw_config *config,
                          struct calibration_params *params,
                          struct calibration_options *options) __must_check;

/*
 * Verifies current calibration parameters against reference values
 * @param params Pointer to calibration parameters to verify
 * @param current_temperature Current operating temperature
 * @return 0 if valid, negative error code if recalibration needed
 */
int lidar_verify_calibration(const struct calibration_params *params,
                            float current_temperature) __must_check;

/* Error Codes */
#define LIDAR_CAL_SUCCESS           0   /* Calibration successful */
#define LIDAR_CAL_ERROR_TEMP       -1   /* Temperature out of range */
#define LIDAR_CAL_ERROR_ACCURACY   -2   /* Accuracy requirements not met */
#define LIDAR_CAL_ERROR_TIMEOUT    -3   /* Calibration timeout */
#define LIDAR_CAL_ERROR_HARDWARE   -4   /* Hardware error during calibration */
#define LIDAR_CAL_ERROR_PARAM      -5   /* Invalid parameters */
#define LIDAR_CAL_ERROR_VERIFY     -6   /* Verification failed */

/* Calibration Flags */
#define LIDAR_CAL_FLAG_FORCE       (1 << 0)  /* Force recalibration */
#define LIDAR_CAL_FLAG_QUICK       (1 << 1)  /* Quick calibration mode */
#define LIDAR_CAL_FLAG_VERBOSE     (1 << 2)  /* Verbose output */
#define LIDAR_CAL_FLAG_TEMP_COMP   (1 << 3)  /* Enable temperature compensation */
#define LIDAR_CAL_FLAG_INTENSITY   (1 << 4)  /* Include intensity calibration */
#define LIDAR_CAL_FLAG_VERIFY      (1 << 5)  /* Verify after calibration */

#endif /* _LIDAR_CALIBRATION_H_ */