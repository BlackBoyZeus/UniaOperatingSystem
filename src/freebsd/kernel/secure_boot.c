/**
 * @file secure_boot.c
 * @brief Implementation of secure boot subsystem for TALD UNIA's FreeBSD-based OS
 * @version 1.0
 */

#include "secure_boot.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

/* TPM2-TSS v3.2.0 */
#include <tss2/tss2_esys.h>

/* UEFI v2.9 */
#include <efi.h>

/* Global variables */
static pthread_mutex_t g_measurement_mutex = PTHREAD_MUTEX_INITIALIZER;
static SecureBootManager* g_instance = NULL;

/* Static function declarations */
static bool validate_boot_entries(const boot_entry_t entries[], uint32_t count);
static status_t verify_uefi_status(void);
static measurement_t hash_component(const uint8_t* data, size_t size);
static bool verify_signature(const uint8_t* data, size_t size, const uint8_t* signature);
static void log_measurement(measurement_log_t* log, const measurement_t* measurement);

/**
 * @brief Verifies the integrity of the entire boot chain
 */
__attribute__((boot_critical)) __attribute__((measure_performance))
status_t verify_boot_chain(boot_entry_t boot_entries[], uint32_t policy_version, uint64_t timestamp) {
    status_t status = {0};
    
    /* Validate inputs */
    if (!boot_entries || policy_version == 0) {
        status.error_code = SECURE_BOOT_INVALID_PARAMS;
        return status;
    }

    /* Verify UEFI secure boot status */
    status_t uefi_status = verify_uefi_status();
    if (uefi_status.error_code != 0) {
        return uefi_status;
    }

    /* Initialize TPM session */
    TpmManager& tpm = TpmManager::getInstance();
    
    /* Verify each boot component */
    for (int i = 0; i < BOOT_CHAIN_MAX_DEPTH && boot_entries[i].data != NULL; i++) {
        /* Verify component signature */
        if (!verify_signature(boot_entries[i].data, boot_entries[i].size, 
                            boot_entries[i].signature)) {
            status.error_code = SECURE_BOOT_SIGNATURE_INVALID;
            return status;
        }

        /* Measure component into TPM */
        measurement_t measurement = measure_boot_component(
            boot_entries[i].data,
            boot_entries[i].size,
            TPM_PCR_BOOT_CODE,
            boot_entries[i].type
        );

        /* Extend PCR with measurement */
        if (!tpm.extend_pcr(TPM_PCR_BOOT_CODE, measurement.hash)) {
            status.error_code = SECURE_BOOT_TPM_ERROR;
            return status;
        }
    }

    /* Verify policy version and update PCR */
    uint8_t policy_data[sizeof(uint32_t)];
    memcpy(policy_data, &policy_version, sizeof(policy_version));
    
    measurement_t policy_measurement = measure_boot_component(
        policy_data,
        sizeof(policy_data),
        TPM_PCR_BOOT_POLICY,
        MEASUREMENT_TYPE_POLICY
    );

    if (!tpm.extend_pcr(TPM_PCR_BOOT_POLICY, policy_measurement.hash)) {
        status.error_code = SECURE_BOOT_POLICY_ERROR;
        return status;
    }

    status.error_code = SECURE_BOOT_SUCCESS;
    return status;
}

/**
 * @brief Measures a boot component into TPM PCR banks
 */
__attribute__((boot_critical)) __attribute__((atomic))
measurement_t measure_boot_component(const uint8_t* component_data, size_t size,
                                  uint32_t pcr_index, measurement_type_t type,
                                  const char* metadata) {
    measurement_t measurement = {0};
    
    /* Input validation */
    if (!component_data || size == 0 || pcr_index >= MAX_PCR_MEASUREMENTS) {
        measurement.type = MEASUREMENT_TYPE_INVALID;
        return measurement;
    }

    /* Acquire measurement lock */
    pthread_mutex_lock(&g_measurement_mutex);

    /* Calculate hash */
    measurement = hash_component(component_data, size);
    measurement.pcr_index = pcr_index;
    measurement.type = type;
    measurement.timestamp = time(NULL);
    
    if (metadata) {
        strncpy(measurement.metadata, metadata, sizeof(measurement.metadata) - 1);
    }

    /* Log measurement */
    SecureBootManager& manager = SecureBootManager::getInstance();
    log_measurement(manager.get_measurement_log(), &measurement);

    pthread_mutex_unlock(&g_measurement_mutex);
    return measurement;
}

/**
 * @brief SecureBootManager implementation
 */
SecureBootManager::SecureBootManager(const policy_config_t* policy_config,
                                   const tpm_config_t* tpm_config) {
    /* Initialize TPM manager */
    tpm_manager = new TpmManager(tpm_config);
    
    /* Initialize key manager */
    key_manager = new KeyManager();
    
    /* Initialize measurement log */
    measurements = (measurement_log_t*)calloc(1, sizeof(measurement_log_t));
    
    /* Load boot policy */
    boot_policy = (boot_policy_t*)malloc(sizeof(boot_policy_t));
    memcpy(boot_policy, policy_config->boot_policy, sizeof(boot_policy_t));
    
    /* Initialize runtime state */
    runtime_state = (runtime_state_t*)calloc(1, sizeof(runtime_state_t));
}

SecureBootManager::~SecureBootManager() {
    delete tpm_manager;
    delete key_manager;
    free(measurements);
    free(boot_policy);
    free(runtime_state);
}

SecureBootManager& SecureBootManager::getInstance(
    const policy_config_t* policy_config,
    const tpm_config_t* tpm_config) {
    if (!g_instance) {
        g_instance = new SecureBootManager(policy_config, tpm_config);
    }
    return *g_instance;
}

attestation_t SecureBootManager::verify_system_state(
    const uint8_t* nonce,
    uint32_t policy_version) {
    attestation_t result = {0};
    
    /* Collect current PCR values */
    TPML_PCR_SELECTION pcr_selection = {
        .count = 3,
        .pcrSelections = {
            { .hash = TPM2_ALG_SHA384,
              .pcrSelect = {TPM_PCR_BOOT_POLICY, TPM_PCR_BOOT_CODE, TPM_PCR_KERNEL} }
        }
    };

    /* Generate attestation quote */
    result = tpm_manager->attest_system_state(&pcr_selection, nonce);
    
    /* Verify measurement log integrity */
    if (!verify_measurement_log()) {
        result.status = ATTESTATION_LOG_INVALID;
        return result;
    }

    /* Verify policy version */
    if (policy_version != boot_policy->version) {
        result.status = ATTESTATION_POLICY_MISMATCH;
        return result;
    }

    result.status = ATTESTATION_SUCCESS;
    return result;
}

/* Static helper functions */
static bool validate_boot_entries(const boot_entry_t entries[], uint32_t count) {
    if (!entries || count > BOOT_CHAIN_MAX_DEPTH) {
        return false;
    }
    
    for (uint32_t i = 0; i < count; i++) {
        if (!entries[i].data || entries[i].size == 0) {
            return false;
        }
    }
    return true;
}

static status_t verify_uefi_status(void) {
    status_t status = {0};
    EFI_SYSTEM_TABLE* st = NULL;
    
    /* Get UEFI system table */
    if (EFI_ERROR(EFI_GetSystemTable(&st))) {
        status.error_code = SECURE_BOOT_UEFI_ERROR;
        return status;
    }

    /* Verify secure boot is enabled */
    if (!st->SecureBoot || *st->SecureBoot != 1) {
        status.error_code = SECURE_BOOT_DISABLED;
        return status;
    }

    status.error_code = SECURE_BOOT_SUCCESS;
    return status;
}

static measurement_t hash_component(const uint8_t* data, size_t size) {
    measurement_t measurement = {0};
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    
    EVP_DigestInit_ex(ctx, EVP_sha384(), NULL);
    EVP_DigestUpdate(ctx, data, size);
    EVP_DigestFinal_ex(ctx, measurement.hash, NULL);
    
    EVP_MD_CTX_free(ctx);
    return measurement;
}

static bool verify_signature(const uint8_t* data, size_t size, const uint8_t* signature) {
    KeyManager& key_manager = KeyManager::getInstance();
    return key_manager.verify_signature(data, size, signature);
}

static void log_measurement(measurement_log_t* log, const measurement_t* measurement) {
    if (log->count < MAX_PCR_MEASUREMENTS) {
        memcpy(&log->entries[log->count++], measurement, sizeof(measurement_t));
    }
}