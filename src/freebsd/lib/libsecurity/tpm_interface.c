/**
 * @file tpm_interface.c
 * @brief TPM 2.0 interface implementation for TALD UNIA security subsystem
 * @version 1.0
 * @copyright TALD UNIA Platform
 * 
 * Provides hardware-backed security operations, platform attestation, and secure
 * key management with comprehensive error handling and security monitoring.
 */

#include <tss2/tss2_esys.h>  // v3.2.0
#include <tss2/tss2_mu.h>    // v3.2.0
#include <string.h>
#include <stdlib.h>
#include "../config/security.conf"

/* Global state management with secure memory handling */
static ESYS_CONTEXT* g_tpm_context = NULL;
static session_map_t* g_active_sessions = NULL;
static pcr_bank_t g_pcr_banks[24];
static key_cache_t* g_key_cache = NULL;
static tpm_health_t g_tpm_health;

/* Security-critical operation tracking */
#define SECURITY_CRITICAL __attribute__((section(".secure_text")))
#define HARDWARE_REQUIRED __attribute__((warn_unused_result))

/* Secure memory handling macros */
#define SECURE_MEMZERO(ptr, size) do { \
    volatile unsigned char *vptr = (volatile unsigned char *)(ptr); \
    size_t i; \
    for (i = 0; i < (size); i++) \
        vptr[i] = 0; \
} while (0)

/**
 * @brief TPM error handling with secure logging
 */
static void handle_tpm_error(TSS2_RC rc, const char* operation) {
    if (rc != TSS2_RC_SUCCESS) {
        // Log error securely with sanitized output
        secure_log(SECURITY_CRITICAL_LEVEL, 
                  "TPM operation failed: %s (code: 0x%X)", 
                  operation, rc);
        
        // Update TPM health status
        g_tpm_health.last_error = rc;
        g_tpm_health.error_count++;
        
        // Check for critical failure conditions
        if (g_tpm_health.error_count >= TPM_MAX_ERRORS) {
            trigger_security_lockdown();
        }
    }
}

/**
 * @brief Initialize TPM device with comprehensive checks
 */
SECURITY_CRITICAL HARDWARE_REQUIRED
tpm_status_t tpm_initialize(void) {
    TSS2_RC rc;
    tpm_status_t status = {0};
    
    // Verify TPM version compatibility
    if (TPM_VERSION != 2.0) {
        status.code = TPM_ERROR_VERSION_MISMATCH;
        return status;
    }
    
    // Initialize ESYS context with enhanced security
    rc = Esys_Initialize(&g_tpm_context, NULL, NULL);
    if (rc != TSS2_RC_SUCCESS) {
        handle_tpm_error(rc, "Esys_Initialize");
        status.code = TPM_ERROR_INIT_FAILED;
        return status;
    }
    
    // Perform TPM startup with health checks
    rc = Esys_Startup(g_tpm_context, TPM2_SU_CLEAR);
    if (rc != TSS2_RC_SUCCESS) {
        handle_tpm_error(rc, "Esys_Startup");
        status.code = TPM_ERROR_STARTUP_FAILED;
        return status;
    }
    
    // Initialize secure session management
    g_active_sessions = session_map_create();
    if (!g_active_sessions) {
        status.code = TPM_ERROR_SESSION_INIT_FAILED;
        return status;
    }
    
    // Setup PCR banks with measurement tracking
    memset(g_pcr_banks, 0, sizeof(g_pcr_banks));
    rc = initialize_pcr_banks(g_tpm_context, g_pcr_banks);
    if (rc != TSS2_RC_SUCCESS) {
        handle_tpm_error(rc, "PCR_Initialize");
        status.code = TPM_ERROR_PCR_INIT_FAILED;
        return status;
    }
    
    // Initialize key cache with rotation policies
    g_key_cache = key_cache_create();
    if (!g_key_cache) {
        status.code = TPM_ERROR_KEY_CACHE_INIT_FAILED;
        return status;
    }
    
    // Initialize health monitoring
    g_tpm_health = (tpm_health_t){
        .status = TPM_HEALTH_GOOD,
        .error_count = 0,
        .last_check = time(NULL)
    };
    
    status.code = TPM_SUCCESS;
    return status;
}

/**
 * @brief TpmManager implementation
 */
typedef struct TpmManager {
    ESYS_CONTEXT* esys_context;
    session_map_t* active_sessions;
    pcr_bank_t* pcr_banks;
    nv_index_map_t* nv_indices;
    key_rotation_manager_t* key_manager;
    tpm_health_monitor_t* health_monitor;
} TpmManager;

SECURITY_CRITICAL
static TpmManager* TpmManager_create(void) {
    TpmManager* manager = calloc(1, sizeof(TpmManager));
    if (!manager) return NULL;
    
    // Initialize with global context
    manager->esys_context = g_tpm_context;
    manager->active_sessions = g_active_sessions;
    manager->pcr_banks = g_pcr_banks;
    
    // Setup additional components
    manager->nv_indices = nv_index_map_create();
    manager->key_manager = key_rotation_manager_create();
    manager->health_monitor = tpm_health_monitor_create();
    
    return manager;
}

SECURITY_CRITICAL
static key_handle_t TpmManager_create_primary_key(
    TpmManager* manager,
    const tpm2_template_t* key_template,
    const key_rotation_policy_t* rotation_policy
) {
    if (!manager || !key_template || !rotation_policy) {
        return (key_handle_t){.handle = TPM2_HANDLE_NULL};
    }
    
    TSS2_RC rc;
    TPM2B_PUBLIC* public = NULL;
    TPM2B_PRIVATE* private = NULL;
    ESYS_TR handle = ESYS_TR_NONE;
    
    // Create primary key with template
    rc = Esys_CreatePrimary(
        manager->esys_context,
        ESYS_TR_RH_OWNER,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        &key_template->sensitive,
        &key_template->public,
        &key_template->outside,
        &key_template->creation,
        &handle,
        &public,
        &private,
        NULL,
        NULL
    );
    
    if (rc != TSS2_RC_SUCCESS) {
        handle_tpm_error(rc, "CreatePrimary");
        return (key_handle_t){.handle = TPM2_HANDLE_NULL};
    }
    
    // Apply rotation policy
    key_handle_t key_handle = {
        .handle = handle,
        .creation_time = time(NULL),
        .rotation_due = time(NULL) + rotation_policy->rotation_interval,
        .policy = *rotation_policy
    };
    
    // Cache key with metadata
    key_cache_add(g_key_cache, &key_handle);
    
    // Cleanup sensitive data
    Esys_Free(public);
    Esys_Free(private);
    
    return key_handle;
}

/**
 * @brief Cleanup and shutdown
 */
SECURITY_CRITICAL
void tpm_cleanup(void) {
    if (g_tpm_context) {
        Esys_Finalize(&g_tpm_context);
        g_tpm_context = NULL;
    }
    
    if (g_active_sessions) {
        session_map_destroy(g_active_sessions);
        g_active_sessions = NULL;
    }
    
    if (g_key_cache) {
        key_cache_destroy(g_key_cache);
        g_key_cache = NULL;
    }
    
    // Secure memory cleanup
    SECURE_MEMZERO(g_pcr_banks, sizeof(g_pcr_banks));
    SECURE_MEMZERO(&g_tpm_health, sizeof(g_tpm_health));
}

/* Export TpmManager interface */
const struct TpmManager_vtable TpmManager_interface = {
    .create = TpmManager_create,
    .create_primary_key = TpmManager_create_primary_key,
    .attest_platform = TpmManager_attest_platform,
    .monitor_health = TpmManager_monitor_health
};