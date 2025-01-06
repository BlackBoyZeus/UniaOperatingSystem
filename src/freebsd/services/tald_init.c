/*
 * TALD UNIA Gaming Platform - Initialization Service Implementation
 * Version: 1.0.0
 *
 * Implements the initialization and startup sequence for the TALD UNIA gaming platform,
 * coordinating core system, device management, security services, and thermal management.
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/module.h>
#include <sys/kernel.h>
#include <sys/systm.h>
#include <sys/malloc.h>
#include <sys/mutex.h>

/* Internal headers - v1.0.0 */
#include "tald_init.h"
#include "../kernel/tald_core.h"
#include "device_manager.h"

/* Global state tracking */
static struct tald_init_state {
    uint32_t init_flags;
    struct tald_core_config* core_config;
    DeviceManager* device_manager;
    ThermalMonitor* thermal_monitor;
    PowerManager* power_manager;
    TPMManager* tpm_manager;
    uint8_t boot_status;
    uint32_t uptime_ms;
    struct thermal_state thermal_status;
    struct power_state power_status;
    struct recovery_state recovery_status;
    struct mtx state_lock;
} g_tald_init_state;

/* Static function declarations */
static int validate_init_config(const struct tald_init_config* config);
static int initialize_tpm(const struct tpm_config* config);
static int verify_secure_boot(void);
static int setup_thermal_monitoring(const struct thermal_config* config);
static int setup_power_management(const struct power_config* config);
static int initialize_device_manager(const struct device_manager_config* config);
static int configure_recovery_system(uint32_t recovery_flags);
static void update_boot_status(uint8_t status);

/*
 * Primary initialization function for TALD UNIA platform
 */
__must_check
int tald_init(struct tald_init_config* config) {
    int error;

    /* Initialize state lock */
    mtx_init(&g_tald_init_state.state_lock, "tald_init_lock", NULL, MTX_DEF);
    
    /* Validate initialization configuration */
    error = validate_init_config(config);
    if (error != 0) {
        printf("TALD: Invalid initialization configuration\n");
        return TALD_INIT_ERR_CONFIG;
    }

    /* Initialize TPM and perform hardware attestation */
    error = initialize_tpm(config->tpm_config);
    if (error != 0) {
        printf("TALD: TPM initialization failed\n");
        return TALD_INIT_ERR_TPM;
    }

    /* Verify secure boot sequence */
    error = verify_secure_boot();
    if (error != 0) {
        printf("TALD: Secure boot verification failed\n");
        return TALD_INIT_ERR_SECURITY;
    }

    /* Initialize core system */
    error = tald_core_init(config->core_config);
    if (error != 0) {
        printf("TALD: Core system initialization failed\n");
        return TALD_INIT_ERR_CORE;
    }
    g_tald_init_state.core_config = config->core_config;

    /* Setup thermal monitoring */
    error = setup_thermal_monitoring(config->thermal_config);
    if (error != 0) {
        printf("TALD: Thermal monitoring setup failed\n");
        return TALD_INIT_ERR_THERMAL;
    }

    /* Setup power management */
    error = setup_power_management(config->power_config);
    if (error != 0) {
        printf("TALD: Power management setup failed\n");
        return TALD_INIT_ERR_CORE;
    }

    /* Initialize device manager */
    error = initialize_device_manager(config->device_config);
    if (error != 0) {
        printf("TALD: Device manager initialization failed\n");
        return TALD_INIT_ERR_DEVICE;
    }

    /* Configure recovery system */
    error = configure_recovery_system(config->recovery_flags);
    if (error != 0) {
        printf("TALD: Recovery system configuration failed\n");
        return TALD_INIT_ERR_RECOVERY;
    }

    /* Start system monitoring */
    g_tald_init_state.init_flags = config->flags;
    g_tald_init_state.uptime_ms = 0;
    update_boot_status(STAGE_SERVICES);

    printf("TALD: System initialization complete\n");
    return TALD_INIT_SUCCESS;
}

/*
 * Coordinated shutdown of TALD UNIA platform
 */
void tald_shutdown(void) {
    mtx_lock(&g_tald_init_state.state_lock);

    /* Preserve system state */
    if (g_tald_init_state.init_flags & RECOVERY_PRESERVE_STATE) {
        /* Save current state for recovery */
        g_tald_init_state.recovery_status.preserve_state();
    }

    /* Stop monitoring services */
    if (g_tald_init_state.thermal_monitor) {
        g_tald_init_state.thermal_monitor->shutdown();
    }

    /* Transition power states */
    if (g_tald_init_state.power_manager) {
        g_tald_init_state.power_manager->prepare_shutdown();
    }

    /* Shutdown device manager */
    if (g_tald_init_state.device_manager) {
        device_manager_shutdown();
    }

    /* Save TPM state */
    if (g_tald_init_state.tpm_manager) {
        g_tald_init_state.tpm_manager->preserve_state();
    }

    /* Cleanup core system */
    if (g_tald_init_state.core_config) {
        tald_core_shutdown();
    }

    mtx_unlock(&g_tald_init_state.state_lock);
    mtx_destroy(&g_tald_init_state.state_lock);

    printf("TALD: System shutdown complete\n");
}

/* Static function implementations */

static int validate_init_config(const struct tald_init_config* config) {
    if (!config || !config->core_config || !config->thermal_config || 
        !config->power_config || !config->tpm_config) {
        return EINVAL;
    }

    if (config->version != TALD_INIT_VERSION) {
        return EINVAL;
    }

    return 0;
}

static int initialize_tpm(const struct tpm_config* config) {
    g_tald_init_state.tpm_manager = TPMManager::getInstance(config);
    if (!g_tald_init_state.tpm_manager) {
        return ENOMEM;
    }

    return g_tald_init_state.tpm_manager->initialize();
}

static int verify_secure_boot(void) {
    if (!g_tald_init_state.tpm_manager) {
        return EINVAL;
    }

    return g_tald_init_state.tpm_manager->verify_secure_boot();
}

static int setup_thermal_monitoring(const struct thermal_config* config) {
    g_tald_init_state.thermal_monitor = ThermalMonitor::getInstance(config);
    if (!g_tald_init_state.thermal_monitor) {
        return ENOMEM;
    }

    return g_tald_init_state.thermal_monitor->initialize();
}

static int setup_power_management(const struct power_config* config) {
    g_tald_init_state.power_manager = PowerManager::getInstance(config);
    if (!g_tald_init_state.power_manager) {
        return ENOMEM;
    }

    return g_tald_init_state.power_manager->initialize();
}

static int initialize_device_manager(const struct device_manager_config* config) {
    g_tald_init_state.device_manager = DeviceManager::getInstance(
        config,
        &g_tald_init_state.thermal_status,
        g_tald_init_state.power_manager->get_config()
    );
    
    if (!g_tald_init_state.device_manager) {
        return ENOMEM;
    }

    return device_manager_init(config, 
                             &g_tald_init_state.thermal_status,
                             g_tald_init_state.power_manager->get_config());
}

static int configure_recovery_system(uint32_t recovery_flags) {
    g_tald_init_state.recovery_status.flags = recovery_flags;
    g_tald_init_state.recovery_status.initialize();
    return 0;
}

static void update_boot_status(uint8_t status) {
    mtx_lock(&g_tald_init_state.state_lock);
    g_tald_init_state.boot_status = status;
    mtx_unlock(&g_tald_init_state.state_lock);
}