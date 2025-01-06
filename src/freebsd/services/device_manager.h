/**
 * @file device_manager.h
 * @brief Device management interface for TALD UNIA gaming platform
 * @version 1.0.0
 *
 * Provides comprehensive device management capabilities including hardware 
 * initialization, state management, security, power management, and thermal
 * monitoring for the TALD UNIA gaming platform.
 */

#ifndef TALD_DEVICE_MANAGER_H
#define TALD_DEVICE_MANAGER_H

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/param.h>
#include <sys/module.h>

/* Internal Headers */
#include "../kernel/tald_core.h"
#include "../lib/libsecurity/key_management.h"

/* Version Information */
#define DEVICE_MANAGER_VERSION "1.0.0"

/* System Constants */
#define MAX_DEVICES 32
#define THERMAL_MONITOR_INTERVAL_MS 100
#define POWER_STATE_TRANSITION_TIMEOUT_MS 1000
#define DEVICE_WATCHDOG_TIMEOUT_MS 5000
#define SECURITY_TOKEN_SIZE 256
#define MAX_RETRY_ATTEMPTS 3

/* Power States */
enum device_power_state {
    POWER_SAVE,        /* Low power consumption mode */
    POWER_BALANCED,    /* Balanced performance/power mode */
    POWER_PERFORMANCE, /* Maximum performance mode */
    POWER_EMERGENCY    /* Emergency power saving mode */
};

/* Device Status Flags */
enum device_status {
    DEVICE_READY = 0x01,           /* Device is operational */
    DEVICE_ERROR = 0x02,           /* Device encountered an error */
    DEVICE_BUSY = 0x04,            /* Device is processing */
    DEVICE_SLEEP = 0x08,           /* Device is in sleep mode */
    DEVICE_THERMAL_WARNING = 0x10  /* Device temperature warning */
};

/* Thermal Thresholds */
enum thermal_threshold {
    TEMP_NORMAL = 60,     /* Normal operating temperature */
    TEMP_WARNING = 75,    /* Temperature warning threshold */
    TEMP_CRITICAL = 85,   /* Critical temperature threshold */
    TEMP_EMERGENCY = 90   /* Emergency shutdown threshold */
};

/* Power Transition Cost */
enum power_transition_cost {
    COST_LOW = 1,     /* Low impact transition */
    COST_MEDIUM = 2,  /* Medium impact transition */
    COST_HIGH = 3     /* High impact transition */
};

/**
 * @brief Device thermal configuration structure
 */
struct thermal_config {
    uint32_t warning_temp_c;      /* Temperature warning threshold */
    uint32_t critical_temp_c;     /* Critical temperature threshold */
    uint32_t emergency_temp_c;    /* Emergency shutdown temperature */
    uint32_t monitor_interval_ms; /* Monitoring interval */
    uint32_t throttle_temp_c;     /* Throttling temperature */
} __packed;

/**
 * @brief Device security configuration structure
 */
struct security_config {
    KeyManager* key_manager;       /* Security key manager */
    uint32_t token_lifetime_ms;    /* Security token lifetime */
    bool require_authentication;   /* Require device authentication */
    bool enable_secure_boot;      /* Enable secure boot check */
    uint32_t retry_limit;         /* Authentication retry limit */
} __packed;

/**
 * @brief Device manager configuration structure
 */
struct device_manager_config {
    uint32_t version;                    /* Manager version */
    uint32_t max_devices;                /* Maximum managed devices */
    struct tald_core_config* core_config; /* Core system configuration */
    struct thermal_config thermal;        /* Thermal configuration */
    struct security_config security;      /* Security configuration */
    uint32_t watchdog_timeout_ms;        /* Watchdog timeout period */
} __packed;

/**
 * @brief Device handle type
 */
typedef uint32_t device_handle_t;

/**
 * @brief Device information structure
 */
struct device_info {
    char device_id[64];           /* Unique device identifier */
    uint32_t hardware_version;    /* Hardware version */
    uint32_t firmware_version;    /* Firmware version */
    uint32_t capabilities;        /* Device capabilities */
    struct thermal_config thermal; /* Thermal configuration */
    uint32_t power_limit_mw;      /* Power consumption limit */
} __packed;

/**
 * @brief Enhanced device management class with security and monitoring
 */
class DeviceManager {
public:
    /**
     * @brief Get singleton instance of DeviceManager
     * @param config Device manager configuration
     * @param thermal_cfg Thermal configuration
     * @param security_cfg Security configuration
     * @return DeviceManager instance
     */
    static DeviceManager* getInstance(
        struct device_manager_config* config,
        struct thermal_config* thermal_cfg,
        struct security_config* security_cfg
    );

    /**
     * @brief Register and authenticate a new device
     * @param device Device information
     * @param token Security token
     * @return Device handle or error code
     */
    __must_check
    device_handle_t register_device(
        struct device_info* device,
        struct security_token* token
    );

    /**
     * @brief Set device power state with thermal consideration
     * @param handle Device handle
     * @param power_state Target power state
     * @param thermal_limits Thermal constraints
     * @return 0 on success, error code on failure
     */
    __must_check
    int set_device_power_state(
        device_handle_t handle,
        uint8_t power_state,
        struct thermal_constraints* thermal_limits
    );

private:
    /* Private constructor for singleton pattern */
    DeviceManager(
        struct device_manager_config* config,
        struct thermal_config* thermal_cfg,
        struct security_config* security_cfg
    );

    /* Prevent copying */
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    struct device_list* devices;          /* List of managed devices */
    struct tald_core_config* core_config; /* Core system configuration */
    KeyManager* key_manager;              /* Security key manager */
    struct thermal_monitor* thermal_monitor; /* Thermal monitoring system */
    struct power_history* power_history;   /* Power usage history */
    struct security_audit* security_log;   /* Security audit log */
    uint8_t current_power_state;          /* Current power state */
    uint8_t thermal_state;                /* Current thermal state */
};

/**
 * @brief Initialize device manager subsystem
 * @param config Device manager configuration
 * @param thermal_cfg Thermal configuration
 * @param security_cfg Security configuration
 * @return 0 on success, error code on failure
 */
__must_check
int device_manager_init(
    struct device_manager_config* config,
    struct thermal_config* thermal_cfg,
    struct security_config* security_cfg
);

/**
 * @brief Shutdown device manager with state preservation
 */
void device_manager_shutdown(void);

/* Error codes */
#define DEVICE_SUCCESS 0
#define DEVICE_ERROR_INIT -1
#define DEVICE_ERROR_CONFIG -2
#define DEVICE_ERROR_SECURITY -3
#define DEVICE_ERROR_THERMAL -4
#define DEVICE_ERROR_POWER -5
#define DEVICE_ERROR_MEMORY -6
#define DEVICE_ERROR_TIMEOUT -7
#define DEVICE_ERROR_INVALID_HANDLE -8

#endif /* TALD_DEVICE_MANAGER_H */