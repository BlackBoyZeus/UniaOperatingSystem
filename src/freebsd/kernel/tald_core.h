/*
 * TALD UNIA Core Kernel Header
 * Version: 1.0.0
 *
 * Core kernel header defining fundamental system interfaces, structures,
 * and management functions for the TALD UNIA gaming platform with enhanced
 * safety, thermal monitoring, and power management capabilities.
 */

#ifndef _TALD_CORE_H_
#define _TALD_CORE_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/module.h>

/* Internal Headers */
#include "gpu_driver.h"
#include "lidar_driver.h"
#include "mesh_network.h"

/* Version Information */
#define TALD_CORE_VERSION          "1.0.0"

/* System Limits and Constants */
#define TALD_MAX_DEVICES          32
#define TALD_MEMORY_LIMIT         4294967296  /* 4GB */
#define TALD_THERMAL_THRESHOLD    85          /* 85°C */
#define TALD_WATCHDOG_TIMEOUT     5000        /* 5 seconds */

/* Power States */
enum tald_power_state {
    POWER_LOW,
    POWER_BALANCED,
    POWER_PERFORMANCE
};

/* Safety Parameters Structure */
struct tald_safety_params {
    uint32_t thermal_threshold_c;     /* Thermal shutdown threshold */
    uint32_t power_limit_mw;          /* Power consumption limit */
    uint32_t watchdog_timeout_ms;     /* Watchdog timer timeout */
    uint32_t error_threshold;         /* Error count threshold */
    uint32_t safety_flags;            /* Safety feature flags */
} __packed;

/* Thermal Monitoring Structure */
struct tald_thermal_limits {
    uint32_t gpu_temp_limit_c;        /* GPU temperature limit */
    uint32_t lidar_temp_limit_c;      /* LiDAR temperature limit */
    uint32_t soc_temp_limit_c;        /* SoC temperature limit */
    uint32_t throttle_temp_c;         /* Throttling temperature */
    uint32_t shutdown_temp_c;         /* Emergency shutdown temperature */
} __packed;

/* Watchdog Configuration */
struct tald_watchdog_config {
    uint32_t timeout_ms;              /* Watchdog timeout period */
    uint32_t reset_counter;           /* Reset counter */
    uint32_t last_reset_reason;       /* Last reset reason code */
    bool enabled;                     /* Watchdog enabled flag */
} __packed;

/* Core Configuration Structure */
struct tald_core_config {
    uint32_t version;                 /* Core version */
    struct gpu_driver_config* gpu_config;      /* GPU configuration */
    struct lidar_driver_config* lidar_config;  /* LiDAR configuration */
    struct mesh_network_config_t* mesh_config; /* Mesh network configuration */
    uint32_t memory_limit;            /* Memory limit in bytes */
    uint8_t power_state;              /* Current power state */
    struct tald_thermal_limits thermal_config; /* Thermal configuration */
    struct tald_watchdog_config watchdog;      /* Watchdog configuration */
    struct tald_safety_params safety_config;   /* Safety parameters */
} __packed;

/* Safety Feature Flags */
#define TALD_SAFETY_THERMAL_MON   (1 << 0)  /* Thermal monitoring */
#define TALD_SAFETY_POWER_MON     (1 << 1)  /* Power monitoring */
#define TALD_SAFETY_WATCHDOG      (1 << 2)  /* Watchdog timer */
#define TALD_SAFETY_ERROR_CHECK   (1 << 3)  /* Error checking */
#define TALD_SAFETY_MEMORY_PROT   (1 << 4)  /* Memory protection */

/* Error Codes */
#define TALD_SUCCESS              0  /* Operation successful */
#define TALD_ERROR_INIT          -1  /* Initialization error */
#define TALD_ERROR_CONFIG        -2  /* Configuration error */
#define TALD_ERROR_THERMAL       -3  /* Thermal error */
#define TALD_ERROR_POWER         -4  /* Power management error */
#define TALD_ERROR_MEMORY        -5  /* Memory error */
#define TALD_ERROR_WATCHDOG      -6  /* Watchdog error */
#define TALD_ERROR_SAFETY        -7  /* Safety check failed */

/*
 * Initialize TALD UNIA core system with enhanced safety and monitoring
 * @param config Core configuration structure
 * @return 0 on success, error code on failure
 */
__must_check
int tald_core_init(struct tald_core_config* config);

/*
 * Perform clean shutdown of TALD UNIA core system
 */
void tald_core_shutdown(void);

/* Kernel attributes */
#define __kernel_export           __attribute__((visibility("default")))
#define __kernel_packed          __attribute__((packed))

/* Export symbols for kernel module use */
__kernel_export extern const struct tald_core_config* tald_get_default_config(void);
__kernel_export extern int tald_get_thermal_state(struct tald_thermal_limits* limits);
__kernel_export extern int tald_set_power_state(enum tald_power_state state);
__kernel_export extern int tald_get_safety_status(struct tald_safety_params* params);

#endif /* _TALD_CORE_H_ */