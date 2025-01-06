/*
 * TALD UNIA Power Management Subsystem
 * Version: 1.0.0
 *
 * Kernel-level power management header defining interfaces and structures
 * for advanced power state management, thermal control, and battery optimization
 * to achieve 4+ hour battery life while maintaining performance targets.
 */

#ifndef _POWER_MGMT_H_
#define _POWER_MGMT_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/power.h>

/* Internal Headers */
#include "tald_core.h"
#include "gpu_driver.h"

/* Version Information */
#define POWER_MGMT_VERSION "1.0.0"

/* Power States */
#define POWER_STATE_MAX 4

/* Thermal Thresholds (°C) */
#define THERMAL_THRESHOLD_DEFAULT 85
#define THERMAL_THRESHOLD_CRITICAL 95

/* Battery Thresholds (%) */
#define BATTERY_THRESHOLD_LOW 15
#define BATTERY_THRESHOLD_CRITICAL 5

/* Power Management Timing */
#define POWER_TRANSITION_DELAY_MS 100

/* Power State Enumeration */
typedef enum power_state {
    POWER_STATE_EMERGENCY = 0,   /* Emergency power saving mode */
    POWER_STATE_LOW = 1,         /* Maximum power saving */
    POWER_STATE_BALANCED = 2,    /* Balance of performance and power */
    POWER_STATE_PERFORMANCE = 3  /* Maximum performance */
} power_state_t;

/* Power Profile Configuration */
struct power_profile {
    uint32_t target_fps;              /* Target frame rate */
    uint32_t gpu_freq_mhz;           /* GPU frequency target */
    uint32_t cpu_freq_mhz;           /* CPU frequency target */
    uint32_t lidar_scan_freq_hz;     /* LiDAR scanning frequency */
    uint32_t mesh_update_freq_hz;    /* Mesh network update frequency */
    float power_limit_watts;         /* Power consumption limit */
    float thermal_target_c;          /* Target temperature */
} __packed;

/* Thermal Management Configuration */
struct thermal_config {
    uint32_t cpu_temp_limit_c;       /* CPU temperature limit */
    uint32_t gpu_temp_limit_c;       /* GPU temperature limit */
    uint32_t battery_temp_limit_c;   /* Battery temperature limit */
    uint32_t throttle_temp_c;        /* Throttling temperature */
    uint32_t emergency_temp_c;       /* Emergency shutdown temperature */
    uint32_t fan_speed_min_rpm;      /* Minimum fan speed */
    uint32_t fan_speed_max_rpm;      /* Maximum fan speed */
} __packed;

/* Battery Management Configuration */
struct battery_config {
    uint32_t capacity_mah;           /* Battery capacity */
    uint32_t voltage_mv;             /* Battery voltage */
    uint32_t charge_limit_ma;        /* Charging current limit */
    uint32_t discharge_limit_ma;     /* Discharge current limit */
    uint32_t low_threshold_pct;      /* Low battery threshold */
    uint32_t critical_threshold_pct; /* Critical battery threshold */
    uint32_t temp_limit_c;          /* Battery temperature limit */
} __packed;

/* Power Management Configuration */
struct power_mgmt_config {
    uint32_t version;                /* Power management version */
    struct power_profile profiles[POWER_STATE_MAX]; /* Power profiles */
    struct thermal_config thermal;    /* Thermal configuration */
    struct battery_config battery;    /* Battery configuration */
    uint32_t transition_delay_ms;     /* State transition delay */
    uint32_t monitoring_interval_ms;  /* Monitoring interval */
    uint32_t safety_flags;           /* Safety feature flags */
} __packed;

/* Power Management Status */
struct power_mgmt_status {
    power_state_t current_state;     /* Current power state */
    uint32_t battery_level_pct;      /* Battery level percentage */
    float cpu_temp_c;               /* CPU temperature */
    float gpu_temp_c;               /* GPU temperature */
    float battery_temp_c;           /* Battery temperature */
    float current_power_watts;      /* Current power consumption */
    uint32_t throttle_status;       /* Thermal throttling status */
    uint32_t performance_level;     /* Current performance level */
} __packed;

/* Safety Feature Flags */
#define POWER_SAFETY_THERMAL_MON    (1 << 0)  /* Thermal monitoring */
#define POWER_SAFETY_BATTERY_MON    (1 << 1)  /* Battery monitoring */
#define POWER_SAFETY_THROTTLING     (1 << 2)  /* Thermal throttling */
#define POWER_SAFETY_EMERGENCY      (1 << 3)  /* Emergency power saving */
#define POWER_SAFETY_PERFORMANCE    (1 << 4)  /* Performance guarantees */

/* Error Codes */
#define POWER_SUCCESS               0  /* Operation successful */
#define POWER_ERROR_INIT          -1  /* Initialization error */
#define POWER_ERROR_CONFIG        -2  /* Configuration error */
#define POWER_ERROR_THERMAL       -3  /* Thermal error */
#define POWER_ERROR_BATTERY       -4  /* Battery error */
#define POWER_ERROR_STATE         -5  /* State transition error */
#define POWER_ERROR_PERFORMANCE   -6  /* Performance guarantee error */

/*
 * Power Management Class
 * Thread-safe power management implementation with safety features
 */
class PowerManager {
public:
    /*
     * Initialize power manager with configuration
     * @param config Power management configuration
     * @throws std::runtime_error on initialization failure
     */
    PowerManager(const struct power_mgmt_config* config);
    
    /*
     * Destructor ensures clean shutdown
     */
    ~PowerManager();

    /*
     * Set system power state with safety checks
     * @param state Target power state
     * @param flags Transition flags
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int set_power_state(power_state_t state, uint32_t flags);

    /*
     * Get current power management status
     * @param status Output status structure
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int get_status(struct power_mgmt_status* status);

    /*
     * Handle thermal event with emergency response
     * @param temp_c Temperature in Celsius
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int handle_thermal_event(float temp_c);

    /*
     * Activate emergency power saving mode
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int emergency_power_save();

private:
    power_state_t current_state;
    uint32_t battery_level;
    uint32_t thermal_level;
    GPUDriver* gpu_driver;
    struct power_mgmt_config config;
    struct mtx state_mutex;
    
    // Prevent copying
    PowerManager(const PowerManager&) = delete;
    PowerManager& operator=(const PowerManager&) = delete;
};

/*
 * Initialize power management subsystem
 * @param config Power management configuration
 * @return 0 on success, error code on failure
 */
[[nodiscard]] __init
int power_mgmt_init(struct power_mgmt_config* config);

/*
 * Set system power state with safety checks
 * @param state Target power state
 * @param flags Transition flags
 * @return 0 on success, error code on failure
 */
[[nodiscard]] __must_check
int power_mgmt_set_state(power_state_t state, uint32_t flags);

/* Export symbols for kernel module use */
extern "C" {
    __kernel_export extern const struct power_mgmt_config* power_get_default_config(void);
    __kernel_export extern int power_get_status(struct power_mgmt_status* status);
    __kernel_export extern int power_set_profile(power_state_t state, struct power_profile* profile);
}

#endif /* _POWER_MGMT_H_ */