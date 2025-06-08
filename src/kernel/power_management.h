/*
 * UNIA Operating System
 * Power Management for AI Gaming - Header File
 */

#ifndef _UNIA_POWER_MANAGEMENT_H_
#define _UNIA_POWER_MANAGEMENT_H_

#include <sys/types.h>

/* Version information */
#define UNIA_POWER_VERSION_MAJOR 1
#define UNIA_POWER_VERSION_MINOR 0
#define UNIA_POWER_VERSION_PATCH 0
#define UNIA_POWER_VERSION ((UNIA_POWER_VERSION_MAJOR << 16) | \
                           (UNIA_POWER_VERSION_MINOR << 8) | \
                           UNIA_POWER_VERSION_PATCH)

/* Feature flags */
#define UNIA_POWER_FEATURE_DYNAMIC_FREQ  0x00000001
#define UNIA_POWER_FEATURE_CORE_CONTROL  0x00000002
#define UNIA_POWER_FEATURE_THERMAL_MGMT  0x00000004
#define UNIA_POWER_FEATURE_BATTERY_AWARE 0x00000008
#define UNIA_POWER_FEATURE_AI_PRIORITY   0x00000010

/* Power modes */
enum unia_power_mode {
    UNIA_POWER_MODE_PERFORMANCE = 0,
    UNIA_POWER_MODE_BALANCED = 1,
    UNIA_POWER_MODE_POWER_SAVE = 2,
    UNIA_POWER_MODE_ADAPTIVE = 3,
    UNIA_POWER_MODE_COUNT
};

/* Structures */

/* Power management information */
struct unia_power_info {
    uint32_t version;           /* Version number */
    uint32_t features;          /* Supported features */
    uint32_t power_modes;       /* Number of power modes */
    enum unia_power_mode current_mode; /* Current power mode */
};

/* Power profile */
struct unia_power_profile {
    char name[32];              /* Profile name */
    char description[128];      /* Profile description */
    int cpu_min_freq;           /* Minimum CPU frequency (percent) */
    int cpu_max_freq;           /* Maximum CPU frequency (percent) */
    int gpu_min_freq;           /* Minimum GPU frequency (percent) */
    int gpu_max_freq;           /* Maximum GPU frequency (percent) */
    int active_cores;           /* Active CPU cores (percent) */
    int thermal_limit;          /* Thermal limit (percent) */
    int ai_priority;            /* AI task priority (percent) */
    int battery_threshold;      /* Battery threshold for mode switch (percent) */
};

/* Function prototypes */

/* Initialize the power management module */
int unia_power_init(void);

/* Cleanup the power management module */
void unia_power_cleanup(void);

/* Get information about the power management module */
int unia_power_get_info(struct unia_power_info *info);

/* Set the power mode */
int unia_power_set_mode(enum unia_power_mode mode);

/* Get the current power mode */
enum unia_power_mode unia_power_get_mode(void);

/* Get the power profile for a specific mode */
int unia_power_get_profile(enum unia_power_mode mode, struct unia_power_profile *profile);

/* Set the power profile for a specific mode */
int unia_power_set_profile(enum unia_power_mode mode, const struct unia_power_profile *profile);

/* Notify the power management system of AI workload */
int unia_power_notify_ai_workload(int workload_percent);

#endif /* _UNIA_POWER_MANAGEMENT_H_ */
