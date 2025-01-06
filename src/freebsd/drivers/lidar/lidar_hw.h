/*
 * TALD UNIA LiDAR Hardware Interface
 * Version: 1.0.0
 * 
 * Hardware interface header file defining low-level structures and functions
 * for controlling the LiDAR hardware subsystem with:
 * - 30Hz scanning frequency
 * - 0.01cm resolution
 * - 5-meter effective range
 * - Real-time point cloud processing
 * - DMA management
 * - Safety features
 * - Diagnostic capabilities
 */

#ifndef _LIDAR_HW_H_
#define _LIDAR_HW_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/param.h>
#include <sys/module.h>
#include <sys/kernel.h>
#include <sys/bus.h>

/* Version Information */
#define LIDAR_HW_VERSION          "1.0.0"

/* Hardware Specifications */
#define LIDAR_SCAN_FREQ_HZ       30      /* 30Hz scanning frequency */
#define LIDAR_RESOLUTION_MM      0.1     /* 0.01cm resolution */
#define LIDAR_RANGE_MM           5000    /* 5-meter range */
#define LIDAR_BUFFER_SIZE        1048576 /* 1MB scan buffer */
#define LIDAR_DMA_CHANNELS       8       /* Number of DMA channels */
#define LIDAR_DMA_ALIGNMENT      4096    /* 4KB DMA alignment */

/* Safety Parameters */
#define LIDAR_MAX_POWER_MW       500     /* Maximum power consumption */
#define LIDAR_TEMP_THRESHOLD_C   75      /* Temperature threshold */
#define LIDAR_SAFETY_CLASS       1       /* Laser safety class 1M */

/* Hardware Configuration Structure */
struct lidar_hw_config {
    uint32_t scan_frequency_hz;      /* Scanning frequency in Hz */
    uint32_t resolution_mm;          /* Resolution in millimeters */
    uint32_t range_mm;              /* Range in millimeters */
    uint32_t buffer_size;           /* Scan buffer size */
    uint32_t dma_channels;          /* Number of DMA channels */
    uint32_t dma_alignment;         /* DMA buffer alignment */
    uint32_t power_limit_mw;        /* Power limit in milliwatts */
    uint32_t temp_threshold_c;      /* Temperature threshold in Celsius */
    uint32_t safety_features;       /* Safety feature flags */
    uint32_t calibration_version;   /* Calibration data version */
} __packed;

/* Calibration Data Structure */
struct lidar_calibration_data {
    float angle_corrections[360];    /* Angle correction table */
    float distance_corrections[100]; /* Distance correction table */
    uint32_t temperature_coefficients[10]; /* Temperature compensation */
    uint32_t calibration_timestamp;  /* Calibration timestamp */
    uint32_t calibration_checksum;   /* Calibration data checksum */
} __packed;

/* Function Prototypes */

/*
 * Initialize LiDAR hardware subsystem
 * @param config Pointer to hardware configuration
 * @param cal_data Pointer to calibration data
 * @return 0 on success, error code on failure
 */
int lidar_hw_init(struct lidar_hw_config *config, 
                  struct lidar_calibration_data *cal_data) __must_check;

/*
 * Hardware interrupt handler
 * @param context Interrupt context
 */
void lidar_hw_interrupt_handler(void *context) __interrupt;

/* Error Codes */
#define LIDAR_ERR_SUCCESS        0  /* Operation successful */
#define LIDAR_ERR_INVALID_CONFIG 1  /* Invalid configuration */
#define LIDAR_ERR_CALIBRATION    2  /* Calibration error */
#define LIDAR_ERR_HARDWARE       3  /* Hardware failure */
#define LIDAR_ERR_TEMPERATURE    4  /* Temperature threshold exceeded */
#define LIDAR_ERR_POWER         5  /* Power limit exceeded */
#define LIDAR_ERR_DMA           6  /* DMA error */
#define LIDAR_ERR_SAFETY        7  /* Safety interlock triggered */

/* Safety Feature Flags */
#define LIDAR_SAFETY_TEMP_MON   (1 << 0)  /* Temperature monitoring */
#define LIDAR_SAFETY_POWER_MON  (1 << 1)  /* Power monitoring */
#define LIDAR_SAFETY_INTERLOCKS (1 << 2)  /* Safety interlocks */
#define LIDAR_SAFETY_WATCHDOG   (1 << 3)  /* Watchdog timer */
#define LIDAR_SAFETY_ECC        (1 << 4)  /* Error correction */
#define LIDAR_SAFETY_REDUNDANCY (1 << 5)  /* Redundant sensors */

/* DMA Control Flags */
#define LIDAR_DMA_ACTIVE        (1 << 0)  /* DMA channel active */
#define LIDAR_DMA_ERROR         (1 << 1)  /* DMA error occurred */
#define LIDAR_DMA_COMPLETE      (1 << 2)  /* DMA transfer complete */
#define LIDAR_DMA_OVERFLOW      (1 << 3)  /* Buffer overflow */

#endif /* _LIDAR_HW_H_ */