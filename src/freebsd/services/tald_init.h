/*
 * TALD UNIA Gaming Platform - Initialization Service Header
 * Version: 1.0.0
 *
 * Defines the initialization and startup sequence for the TALD UNIA gaming platform,
 * coordinating core system, device management, security services, and thermal management
 * with enhanced recovery capabilities.
 */

#ifndef _TALD_INIT_H_
#define _TALD_INIT_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/param.h>
#include <sys/module.h>

/* Internal Headers */
#include "../kernel/tald_core.h"
#include "device_manager.h"

/* Version Information */
#define TALD_INIT_VERSION "1.0.0"

/* Initialization Flags */
enum tald_init_flags {
    INIT_SECURE = 1,          /* Enable secure initialization */
    INIT_DEBUG = 2,           /* Enable debug mode */
    INIT_RECOVERY = 4,        /* Enable recovery mode */
    INIT_THERMAL_SAFE = 8,    /* Enable thermal safety checks */
    INIT_TPM_REQUIRED = 16    /* Require TPM for initialization */
};

/* Initialization Stages */
enum tald_init_stages {
    STAGE_SECURITY = 1,       /* Security subsystem initialization */
    STAGE_CORE = 2,           /* Core system initialization */
    STAGE_DEVICES = 3,        /* Device initialization */
    STAGE_SERVICES = 4        /* Service initialization */
};

/* Constants */
#define TALD_INIT_TIMEOUT 30000  /* 30 second initialization timeout */

/**
 * Enhanced initialization configuration structure
 */
struct tald_init_config {
    uint32_t version;                         /* Initialization version */
    uint32_t flags;                          /* Initialization flags */
    struct tald_core_config* core_config;     /* Core system configuration */
    struct device_manager_config* device_config; /* Device manager configuration */
    struct thermal_config* thermal_config;    /* Thermal management configuration */
    struct tpm_config* tpm_config;           /* TPM configuration */
    uint32_t timeout_ms;                     /* Initialization timeout */
    uint32_t recovery_flags;                 /* Recovery mode flags */

    /**
     * Initialize configuration with enhanced default values
     */
    void (*init_defaults)(struct tald_init_config* config);

    /**
     * Validate initialization configuration
     * @return true if valid, false otherwise
     */
    bool (*validate)(const struct tald_init_config* config);
} __packed;

/**
 * Enhanced primary initialization function for TALD UNIA platform
 * @param config Initialization configuration
 * @return 0 on success, error code on failure
 */
__must_check __no_interrupt
int tald_init(struct tald_init_config* config);

/**
 * Enhanced coordinated shutdown of TALD UNIA platform
 * @param flags Shutdown flags
 * @return 0 on success, error code on failure
 */
__no_interrupt
int tald_shutdown(uint32_t flags);

/* Error Codes */
#define TALD_INIT_SUCCESS          0   /* Initialization successful */
#define TALD_INIT_ERR_CONFIG      -1   /* Configuration error */
#define TALD_INIT_ERR_SECURITY    -2   /* Security initialization error */
#define TALD_INIT_ERR_CORE        -3   /* Core system initialization error */
#define TALD_INIT_ERR_DEVICE      -4   /* Device initialization error */
#define TALD_INIT_ERR_THERMAL     -5   /* Thermal management error */
#define TALD_INIT_ERR_TPM         -6   /* TPM initialization error */
#define TALD_INIT_ERR_TIMEOUT     -7   /* Initialization timeout */
#define TALD_INIT_ERR_RECOVERY    -8   /* Recovery mode error */

/* Recovery Flags */
#define RECOVERY_PRESERVE_STATE    0x01  /* Preserve system state during recovery */
#define RECOVERY_SAFE_MODE        0x02  /* Initialize in safe mode */
#define RECOVERY_THERMAL_RESET    0x04  /* Reset thermal subsystem */
#define RECOVERY_FACTORY_RESET    0x08  /* Perform factory reset */

#endif /* _TALD_INIT_H_ */