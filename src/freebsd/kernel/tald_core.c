/*
 * TALD UNIA Core Kernel Module
 * Version: 1.0.0
 *
 * Core implementation of the TALD UNIA gaming platform kernel module providing:
 * - System initialization and resource management
 * - Thermal monitoring and throttling
 * - Enhanced GPU, LiDAR, and mesh network coordination
 * - Interrupt-safe operations
 */

#include <sys/types.h>      // v9.0
#include <sys/param.h>      // v9.0
#include <sys/kernel.h>     // v9.0
#include <sys/module.h>     // v9.0
#include <sys/systm.h>      // v9.0
#include <sys/interrupt.h>  // v9.0

#include "tald_core.h"
#include "gpu_driver.h"
#include "lidar_driver.h"
#include "mesh_network.h"

/* Global state */
static struct tald_core_state* g_tald_core = NULL;
static struct mtx g_core_lock;
static struct thermal_state g_thermal_state;

/* Thermal state structure */
struct thermal_state {
    uint32_t gpu_temp_c;
    uint32_t lidar_temp_c;
    uint32_t soc_temp_c;
    uint32_t throttle_level;
    uint64_t last_update;
};

/* Core state structure */
struct tald_core_state {
    struct tald_core_config config;
    struct GPUDriver* gpu_driver;
    struct lidar_driver_state* lidar_state;
    struct mesh_fleet_t* mesh_fleet;
    struct thermal_state* thermal_monitor;
    uint8_t system_status;
    uint8_t current_power_state;
    uint8_t thermal_throttle_level;
};

/* Forward declarations */
static int init_thermal_monitoring(struct tald_core_state* core);
static int init_subsystems(struct tald_core_state* core);
static void thermal_monitor_callback(void* context);

/*
 * Initialize TALD UNIA core system
 */
int
tald_core_init(struct tald_core_config* config)
{
    int error;

    /* Parameter validation */
    if (config == NULL || config->version != TALD_CORE_VERSION) {
        return TALD_ERROR_CONFIG;
    }

    /* Initialize core mutex */
    mtx_init(&g_core_lock, "tald_core_lock", NULL, MTX_DEF | MTX_SPIN);

    /* Allocate core state */
    g_tald_core = malloc(sizeof(struct tald_core_state), M_DEVBUF, M_WAITOK | M_ZERO);
    if (g_tald_core == NULL) {
        mtx_destroy(&g_core_lock);
        return TALD_ERROR_MEMORY;
    }

    /* Copy configuration */
    memcpy(&g_tald_core->config, config, sizeof(struct tald_core_config));

    /* Initialize thermal monitoring */
    error = init_thermal_monitoring(g_tald_core);
    if (error != 0) {
        free(g_tald_core, M_DEVBUF);
        mtx_destroy(&g_core_lock);
        return error;
    }

    /* Initialize subsystems */
    error = init_subsystems(g_tald_core);
    if (error != 0) {
        free(g_tald_core, M_DEVBUF);
        mtx_destroy(&g_core_lock);
        return error;
    }

    /* Setup thermal monitoring callback */
    callout_init(&g_thermal_state.callout, CALLOUT_MPSAFE);
    callout_reset(&g_thermal_state.callout, hz/10, thermal_monitor_callback, NULL);

    return TALD_SUCCESS;
}

/*
 * Initialize thermal monitoring subsystem
 */
static int
init_thermal_monitoring(struct tald_core_state* core)
{
    /* Initialize thermal state */
    core->thermal_monitor = &g_thermal_state;
    core->thermal_monitor->gpu_temp_c = 0;
    core->thermal_monitor->lidar_temp_c = 0;
    core->thermal_monitor->soc_temp_c = 0;
    core->thermal_monitor->throttle_level = 0;
    core->thermal_monitor->last_update = 0;

    /* Set initial thermal limits */
    if (core->config.thermal_config.gpu_temp_limit_c == 0) {
        core->config.thermal_config.gpu_temp_limit_c = TALD_THERMAL_THRESHOLD;
    }
    if (core->config.thermal_config.lidar_temp_limit_c == 0) {
        core->config.thermal_config.lidar_temp_limit_c = LIDAR_THERMAL_SHUTDOWN_C;
    }

    return TALD_SUCCESS;
}

/*
 * Initialize core subsystems
 */
static int
init_subsystems(struct tald_core_state* core)
{
    int error;

    /* Initialize GPU subsystem */
    core->gpu_driver = new GPUDriver(core->config.gpu_config, NULL);
    if (core->gpu_driver == NULL) {
        return TALD_ERROR_INIT;
    }

    /* Initialize LiDAR subsystem */
    error = lidar_driver_init(core->config.lidar_config, &core->config.safety_config);
    if (error != 0) {
        delete core->gpu_driver;
        return error;
    }

    /* Initialize mesh networking */
    core->mesh_fleet = mesh_network_create_fleet("primary", NULL, M_DEVBUF);
    if (core->mesh_fleet == NULL) {
        delete core->gpu_driver;
        return TALD_ERROR_INIT;
    }

    return TALD_SUCCESS;
}

/*
 * Thermal monitoring interrupt handler
 */
void
tald_core_thermal_monitor(void)
{
    struct thermal_state* thermal;
    gpu_power_metrics_t gpu_metrics;
    struct lidar_thermal_state lidar_thermal;
    uint32_t new_throttle_level = 0;

    mtx_lock_spin(&g_core_lock);
    thermal = g_tald_core->thermal_monitor;

    /* Get GPU thermal state */
    if (g_tald_core->gpu_driver->get_power_metrics(&gpu_metrics) == 0) {
        thermal->gpu_temp_c = (uint32_t)gpu_metrics.current_temp_celsius;
    }

    /* Get LiDAR thermal state */
    if (lidar_get_thermal_state(&lidar_thermal) == 0) {
        thermal->lidar_temp_c = lidar_thermal.current_temp_c;
    }

    /* Check thermal thresholds */
    if (thermal->gpu_temp_c >= g_tald_core->config.thermal_config.gpu_temp_limit_c ||
        thermal->lidar_temp_c >= g_tald_core->config.thermal_config.lidar_temp_limit_c) {
        new_throttle_level = 2; /* Severe throttling */
    } else if (thermal->gpu_temp_c >= g_tald_core->config.thermal_config.throttle_temp_c) {
        new_throttle_level = 1; /* Moderate throttling */
    }

    /* Apply throttling if needed */
    if (new_throttle_level != thermal->throttle_level) {
        thermal->throttle_level = new_throttle_level;
        g_tald_core->gpu_driver->set_power_state(
            new_throttle_level == 2 ? GPU_POWER_SAVE :
            new_throttle_level == 1 ? GPU_POWER_BALANCED : GPU_POWER_PERFORMANCE,
            NULL
        );
    }

    thermal->last_update = ticks;
    mtx_unlock_spin(&g_core_lock);
}

/*
 * Thermal monitoring callback
 */
static void
thermal_monitor_callback(void* context)
{
    tald_core_thermal_monitor();
    callout_reset(&g_thermal_state.callout, hz/10, thermal_monitor_callback, NULL);
}

/* Module load/unload handlers */
static int
tald_core_loader(module_t mod, int cmd, void* arg)
{
    int error = 0;

    switch (cmd) {
        case MOD_LOAD:
            printf("TALD UNIA Core: Loading kernel module v%s\n", TALD_CORE_VERSION);
            break;
        case MOD_UNLOAD:
            printf("TALD UNIA Core: Unloading kernel module\n");
            if (g_tald_core != NULL) {
                callout_drain(&g_thermal_state.callout);
                delete g_tald_core->gpu_driver;
                free(g_tald_core, M_DEVBUF);
                mtx_destroy(&g_core_lock);
            }
            break;
        default:
            error = EOPNOTSUPP;
            break;
    }

    return error;
}

/* Module definition */
static moduledata_t tald_core_mod = {
    "tald_core",
    tald_core_loader,
    NULL
};

DECLARE_MODULE(tald_core, tald_core_mod, SI_SUB_DRIVERS, SI_ORDER_FIRST);
MODULE_VERSION(tald_core, 1);