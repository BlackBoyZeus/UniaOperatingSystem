/**
 * @file device_manager.c
 * @brief Implementation of TALD UNIA device manager service
 * @version 1.0.0
 *
 * Provides comprehensive device management with thermal monitoring,
 * power optimization, and TPM-backed security features.
 */

#include <sys/types.h>  // FreeBSD 9.0
#include <sys/param.h>  // FreeBSD 9.0
#include <sys/module.h> // FreeBSD 9.0
#include "device_manager.h"
#include "../kernel/tald_core.h"
#include "../lib/libsecurity/key_management.h"

/* Global state */
static DeviceManager* g_device_manager = NULL;
static atomic_t g_device_count = 0;
static struct thermal_monitor* g_thermal_monitor = NULL;

/* Local function prototypes */
static int init_thermal_monitoring(struct thermal_config* config);
static int init_security_context(struct security_config* config);
static int validate_device_config(struct device_manager_config* config);
static void cleanup_device_manager(void);

/**
 * Initialize device manager subsystem
 * @param config Device manager configuration
 * @return 0 on success, error code on failure
 */
__must_check
int device_manager_init(struct device_manager_config* config) {
    int ret;

    /* Validate configuration */
    ret = validate_device_config(config);
    if (ret != DEVICE_SUCCESS) {
        return ret;
    }

    /* Initialize TPM and security context */
    KeyManager* key_manager = KeyManager_getInstance(config->security);
    if (!key_manager) {
        return DEVICE_ERROR_SECURITY;
    }

    /* Initialize thermal monitoring */
    ret = init_thermal_monitoring(&config->thermal);
    if (ret != DEVICE_SUCCESS) {
        cleanup_device_manager();
        return ret;
    }

    /* Create device manager instance */
    g_device_manager = new DeviceManager(config);
    if (!g_device_manager) {
        cleanup_device_manager();
        return DEVICE_ERROR_INIT;
    }

    return DEVICE_SUCCESS;
}

/**
 * Shutdown device manager with cleanup
 */
void device_manager_shutdown(void) {
    if (g_device_manager) {
        /* Save device states */
        for (uint32_t i = 0; i < g_device_count; i++) {
            g_device_manager->set_device_power_state(i, POWER_SAVE);
        }

        /* Cleanup security context */
        if (g_device_manager->key_manager) {
            delete g_device_manager->key_manager;
        }

        /* Cleanup thermal monitoring */
        if (g_thermal_monitor) {
            delete g_thermal_monitor;
            g_thermal_monitor = NULL;
        }

        delete g_device_manager;
        g_device_manager = NULL;
    }

    atomic_set(&g_device_count, 0);
}

/**
 * DeviceManager implementation
 */
DeviceManager::DeviceManager(struct device_manager_config* config) 
    : devices(NULL),
      core_config(NULL),
      key_manager(NULL),
      thermal_monitor(NULL),
      current_power_state(POWER_BALANCED),
      thermal_state({0}) {

    /* Initialize device list */
    devices = (struct device_list*)malloc(sizeof(struct device_list) * config->max_devices);
    if (!devices) {
        return;
    }

    /* Store core configuration */
    core_config = config->core_config;

    /* Initialize TPM-backed key manager */
    key_manager = KeyManager_getInstance(&config->security);
    if (!key_manager) {
        free(devices);
        return;
    }

    /* Setup thermal monitoring */
    thermal_monitor = g_thermal_monitor;
    thermal_state.warning_temp_c = config->thermal.warning_temp_c;
    thermal_state.critical_temp_c = config->thermal.critical_temp_c;
    thermal_state.emergency_temp_c = config->thermal.emergency_temp_c;
}

/**
 * Register new device with security validation
 */
device_handle_t DeviceManager::register_device(struct device_info* device) {
    if (!device || atomic_read(&g_device_count) >= core_config->max_devices) {
        return DEVICE_ERROR_CONFIG;
    }

    /* Validate TPM state */
    if (!key_manager->validateTPM(NULL)) {
        return DEVICE_ERROR_SECURITY;
    }

    /* Generate device security key */
    enhanced_key_spec_t key_spec = {
        .type = KEY_TYPE_AES_256_GCM,
        .attributes = {
            .lifetime_days = 90,
            .hardware_backed = true,
            .exportable = false,
            .requires_authorization = true
        }
    };

    key_handle_t key = key_manager->createKey(key_spec, NULL);
    if (key < 0) {
        return DEVICE_ERROR_SECURITY;
    }

    /* Initialize thermal monitoring */
    device->thermal = thermal_state;

    /* Add to device list */
    uint32_t handle = atomic_inc_return(&g_device_count) - 1;
    memcpy(&devices[handle], device, sizeof(struct device_info));

    return handle;
}

/**
 * Set device power state with thermal awareness
 */
int DeviceManager::set_device_power_state(device_handle_t handle, uint8_t power_state) {
    if (handle >= atomic_read(&g_device_count)) {
        return DEVICE_ERROR_INVALID_HANDLE;
    }

    /* Check thermal conditions */
    struct thermal_state current_thermal;
    thermal_monitor->get_thermal_state(&current_thermal);

    /* Apply thermal-aware power management */
    if (current_thermal.current_temp_c >= thermal_state.warning_temp_c) {
        /* Force lower power state on high temperature */
        power_state = POWER_SAVE;
    }

    /* Update device power state */
    devices[handle].power_limit_mw = 
        power_state == POWER_SAVE ? core_config->safety_config.power_limit_mw / 2 :
        power_state == POWER_BALANCED ? core_config->safety_config.power_limit_mw :
        core_config->safety_config.power_limit_mw * 1.2;

    return DEVICE_SUCCESS;
}

/* Local function implementations */

static int init_thermal_monitoring(struct thermal_config* config) {
    if (!config) {
        return DEVICE_ERROR_CONFIG;
    }

    g_thermal_monitor = new struct thermal_monitor();
    if (!g_thermal_monitor) {
        return DEVICE_ERROR_INIT;
    }

    g_thermal_monitor->warning_temp_c = config->warning_temp_c;
    g_thermal_monitor->critical_temp_c = config->critical_temp_c;
    g_thermal_monitor->emergency_temp_c = config->emergency_temp_c;
    g_thermal_monitor->monitor_interval_ms = config->monitor_interval_ms;

    return DEVICE_SUCCESS;
}

static int init_security_context(struct security_config* config) {
    if (!config) {
        return DEVICE_ERROR_CONFIG;
    }

    /* Validate TPM manufacturer */
    tpm_manufacturer_info_t tpm_info;
    if (!KeyManager_validateTPM(NULL, &tpm_info)) {
        return DEVICE_ERROR_SECURITY;
    }

    return DEVICE_SUCCESS;
}

static int validate_device_config(struct device_manager_config* config) {
    if (!config || !config->core_config) {
        return DEVICE_ERROR_CONFIG;
    }

    if (config->max_devices > TALD_MAX_DEVICES ||
        config->thermal.critical_temp_c > TALD_THERMAL_THRESHOLD) {
        return DEVICE_ERROR_CONFIG;
    }

    return DEVICE_SUCCESS;
}

static void cleanup_device_manager(void) {
    if (g_thermal_monitor) {
        delete g_thermal_monitor;
        g_thermal_monitor = NULL;
    }

    atomic_set(&g_device_count, 0);
}