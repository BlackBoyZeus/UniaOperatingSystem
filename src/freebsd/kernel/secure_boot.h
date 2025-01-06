/**
 * @file secure_boot.h
 * @brief Secure boot implementation for TALD UNIA's FreeBSD-based operating system
 * @version 1.0
 * 
 * Provides hardware-backed boot integrity verification, measurement, and attestation
 * using TPM 2.0 with comprehensive boot chain validation and runtime monitoring.
 */

#ifndef TALD_SECURE_BOOT_H
#define TALD_SECURE_BOOT_H

#include <stdint.h>
#include <stdbool.h>
#include "../../lib/libsecurity/tpm_interface.h" // TPM 2.0 operations - v3.2.0
#include "../../lib/libsecurity/key_management.h" // Key management - v1.0
#include <efi.h> // UEFI boot services - v2.9

#ifdef __cplusplus
extern "C" {
#endif

/* Version and capability constants */
#define SECURE_BOOT_VERSION "1.0"
#define MAX_PCR_MEASUREMENTS 32
#define BOOT_CHAIN_MAX_DEPTH 5
#define MEASUREMENT_LOG_SIZE 4096
#define MAX_SIGNATURE_SIZE 512
#define BOOT_POLICY_VERSION "1.0"

/* TPM PCR allocations */
#define TPM_PCR_BOOT_POLICY 0
#define TPM_PCR_BOOT_CODE 1
#define TPM_PCR_CONFIG 2

/* Type definitions */
typedef enum {
    MEASUREMENT_TYPE_UEFI = 0,
    MEASUREMENT_TYPE_BOOTLOADER,
    MEASUREMENT_TYPE_KERNEL,
    MEASUREMENT_TYPE_POLICY,
    MEASUREMENT_TYPE_CONFIG
} measurement_type_t;

typedef struct {
    uint8_t hash[48];  // SHA-384 hash
    uint32_t pcr_index;
    measurement_type_t type;
    uint64_t timestamp;
    char metadata[256];
} measurement_t;

typedef struct {
    uint8_t* data;
    size_t size;
    uint8_t signature[MAX_SIGNATURE_SIZE];
    measurement_type_t type;
} boot_entry_t;

typedef struct {
    measurement_t entries[MAX_PCR_MEASUREMENTS];
    uint32_t count;
    uint8_t log_signature[MAX_SIGNATURE_SIZE];
} measurement_log_t;

typedef struct {
    uint32_t version;
    bool require_tpm;
    bool allow_debug;
    uint32_t min_tpm_version;
    uint8_t allowed_pcrs[3];
} boot_policy_t;

/**
 * @brief Verifies the integrity of the entire boot chain
 * @param boot_entries Array of boot components to verify
 * @param policy_version Expected boot policy version
 * @param signature_data Signature data for verification
 * @return Status code indicating verification result
 */
__attribute__((boot_critical)) __attribute__((measure_performance))
status_t verify_boot_chain(
    boot_entry_t boot_entries[],
    uint32_t policy_version,
    const uint8_t* signature_data
);

/**
 * @brief Measures a boot component into TPM PCR banks
 * @param component_data Component data to measure
 * @param size Size of component data
 * @param pcr_index Target PCR index
 * @param component_type Type of component being measured
 * @param metadata Additional measurement metadata
 * @return Measurement result
 */
__attribute__((boot_critical)) __attribute__((atomic))
measurement_t measure_boot_component(
    const uint8_t* component_data,
    size_t size,
    uint32_t pcr_index,
    measurement_type_t component_type,
    const char* metadata
);

/**
 * @brief Core secure boot management class
 */
class SecureBootManager {
public:
    /**
     * @brief Get singleton instance of SecureBootManager
     * @param policy_config Boot policy configuration
     * @param tpm_config TPM configuration
     * @return Reference to singleton instance
     */
    static SecureBootManager& getInstance(
        const policy_config_t* policy_config = nullptr,
        const tpm_config_t* tpm_config = nullptr
    );

    /**
     * @brief Verifies current system state with attestation
     * @param nonce Nonce for freshness
     * @param policy_version Expected policy version
     * @return Attestation result with validation data
     */
    attestation_t verify_system_state(
        const uint8_t* nonce,
        uint32_t policy_version
    );

    /**
     * @brief Retrieves the secure measurement log
     * @return Pointer to measurement log
     */
    const measurement_log_t* get_measurement_log() const;

private:
    TpmManager* tpm_manager;
    KeyManager* key_manager;
    measurement_log_t* measurements;
    boot_policy_t* boot_policy;
    runtime_state_t* runtime_state;
    error_handler_t* error_handler;

    SecureBootManager(const policy_config_t* policy_config,
                     const tpm_config_t* tpm_config);
    ~SecureBootManager();

    /* Prevent copying */
    SecureBootManager(const SecureBootManager&) = delete;
    SecureBootManager& operator=(const SecureBootManager&) = delete;
};

#ifdef __cplusplus
}
#endif

#endif /* TALD_SECURE_BOOT_H */