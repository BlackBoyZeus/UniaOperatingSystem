/*
 * TALD UNIA LiDAR Driver
 * Version: 1.0.0
 *
 * Kernel-level LiDAR driver header defining core interfaces and structures
 * for managing LiDAR hardware operations with:
 * - 30Hz continuous scanning
 * - 0.01cm resolution
 * - 5-meter effective range
 * - Enhanced safety features
 * - Kernel-specific optimizations
 */

#ifndef _LIDAR_DRIVER_H_
#define _LIDAR_DRIVER_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/module.h>
#include <sys/kernel.h>
#include <sys/bus.h>
#include <sys/mutex.h>
#include <sys/systm.h>

/* Internal Headers */
#include "lidar_hw.h"

/* Version Information */
#define LIDAR_DRIVER_VERSION          "1.0.0"

/* System Limits and Constants */
#define LIDAR_MAX_DEVICES             8
#define LIDAR_IRQ_PRIORITY           5
#define LIDAR_DMA_BUFFER_SIZE        1048576
#define LIDAR_DMA_ALIGNMENT          4096
#define LIDAR_MAX_SCAN_FREQUENCY_HZ  30
#define LIDAR_MIN_RESOLUTION_MM      1
#define LIDAR_MAX_RANGE_MM           5000
#define LIDAR_THERMAL_SHUTDOWN_C     85
#define LIDAR_ERROR_TIMEOUT_MS       100

/* Safety Parameters Structure */
struct lidar_safety_params {
    uint32_t laser_power_limit_mw;    /* Maximum laser power in mW */
    uint32_t thermal_shutdown_temp_c;  /* Thermal shutdown temperature */
    uint32_t watchdog_timeout_ms;      /* Watchdog timer timeout */
    uint32_t error_threshold;          /* Error count threshold */
    uint32_t safety_flags;             /* Safety feature flags */
} __packed;

/* Power Management Configuration */
struct lidar_power_config {
    uint32_t power_mode;              /* Current power mode */
    uint32_t sleep_timeout_ms;        /* Sleep mode timeout */
    uint32_t power_limit_mw;          /* Power consumption limit */
    uint32_t thermal_throttle_temp_c;  /* Thermal throttling temperature */
} __packed;

/* Thermal Monitoring State */
struct lidar_thermal_state {
    uint32_t current_temp_c;          /* Current temperature */
    uint32_t max_temp_c;              /* Maximum recorded temperature */
    uint32_t throttle_status;         /* Thermal throttling status */
    uint32_t shutdown_count;          /* Thermal shutdown count */
} __packed;

/* Error Tracking State */
struct lidar_error_state {
    uint32_t error_count;             /* Total error count */
    uint32_t last_error_code;         /* Last error code */
    uint32_t error_timestamp;         /* Last error timestamp */
    uint32_t recovery_attempts;       /* Error recovery attempts */
} __packed;

/* Hardware Status Structure */
struct lidar_hw_status {
    uint32_t device_state;            /* Current device state */
    uint32_t scan_counter;            /* Completed scan counter */
    uint32_t dma_status;             /* DMA channel status */
    uint32_t irq_count;              /* Interrupt count */
    struct lidar_thermal_state thermal;  /* Thermal status */
    struct lidar_error_state errors;    /* Error status */
} __packed;

/* Driver Configuration Structure */
struct lidar_driver_config {
    uint32_t device_id;               /* Unique device identifier */
    uint32_t irq_number;              /* IRQ number */
    uint32_t dma_channel;             /* DMA channel number */
    uint32_t dma_alignment;           /* DMA buffer alignment */
    uint32_t watchdog_timeout_ms;     /* Watchdog timer timeout */
    struct lidar_hw_config hw_config;  /* Hardware configuration */
    struct lidar_safety_params safety;  /* Safety parameters */
    struct lidar_power_config power;    /* Power configuration */
} __packed;

/* Driver State Structure */
struct lidar_driver_state {
    device_t device;                  /* Device handle */
    struct lidar_driver_config config;  /* Driver configuration */
    struct lidar_hw_status hw_status;   /* Hardware status */
    struct lidar_thermal_state thermal;  /* Thermal state */
    struct lidar_error_state errors;     /* Error state */
    void *dma_buffer;                 /* DMA buffer pointer */
    struct mtx state_lock;            /* State mutex */
    struct callout watchdog_timer;    /* Watchdog timer */
    uint32_t error_count;             /* Error counter */
    uint32_t last_error_code;         /* Last error code */
} __packed;

/* Function Prototypes */

/*
 * Initialize the LiDAR driver subsystem with enhanced safety checks
 * @param config Driver configuration
 * @param safety Safety parameters
 * @return 0 on success, error code on failure
 */
int lidar_driver_init(struct lidar_driver_config *config,
                     struct lidar_safety_params *safety) __must_check __no_sleep;

/*
 * Attach LiDAR device to the system with resource management
 * @param dev Device handle
 * @param resources Device resources
 * @return 0 on success, error code on failure
 */
int lidar_driver_attach(device_t dev,
                       struct lidar_resources *resources) __must_check;

/* Error Codes */
#define LIDAR_DRIVER_SUCCESS           0  /* Operation successful */
#define LIDAR_DRIVER_ERR_CONFIG        1  /* Configuration error */
#define LIDAR_DRIVER_ERR_RESOURCE      2  /* Resource allocation error */
#define LIDAR_DRIVER_ERR_HARDWARE      3  /* Hardware error */
#define LIDAR_DRIVER_ERR_SAFETY        4  /* Safety check failed */
#define LIDAR_DRIVER_ERR_THERMAL       5  /* Thermal error */
#define LIDAR_DRIVER_ERR_POWER         6  /* Power management error */
#define LIDAR_DRIVER_ERR_DMA           7  /* DMA error */
#define LIDAR_DRIVER_ERR_TIMEOUT       8  /* Operation timeout */

/* Device States */
#define LIDAR_STATE_UNINITIALIZED      0
#define LIDAR_STATE_INITIALIZED        1
#define LIDAR_STATE_SCANNING           2
#define LIDAR_STATE_ERROR              3
#define LIDAR_STATE_THERMAL_SHUTDOWN   4
#define LIDAR_STATE_POWER_SAVE         5

/* Safety Flags */
#define LIDAR_SAFETY_THERMAL_MON      (1 << 0)
#define LIDAR_SAFETY_POWER_MON        (1 << 1)
#define LIDAR_SAFETY_WATCHDOG         (1 << 2)
#define LIDAR_SAFETY_ERROR_RECOVERY   (1 << 3)
#define LIDAR_SAFETY_DMA_PROTECTION   (1 << 4)

#endif /* _LIDAR_DRIVER_H_ */