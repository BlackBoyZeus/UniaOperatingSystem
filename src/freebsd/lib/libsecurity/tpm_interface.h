/**
 * @file tpm_interface.h
 * @brief Core TPM 2.0 interface for TALD UNIA's security subsystem
 * @version 1.0
 * 
 * Provides comprehensive TPM 2.0 integration for hardware security, attestation,
 * and secure operations in the TALD UNIA gaming platform.
 */

#ifndef TALD_TPM_INTERFACE_H
#define TALD_TPM_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>

/* External TPM2-TSS dependencies - v3.2.0 */
#include <tss2/tss2_esys.h>
#include <tss2/tss2_mu.h>

/* Internal configuration */
#include "../config/security.conf"

#ifdef __cplusplus
extern "C" {
#endif

/* Global constants */
#define TPM_MAX_SESSIONS      32
#define TPM_MAX_HANDLES      64
#define TPM_PCR_BANKS        24
#define TPM_NV_MAX_SIZE      2048
#define TPM_SESSION_TIMEOUT   300
#define TPM_MAX_RETRY_COUNT  3
#define TPM_CAPABILITY_MASK  0xFFFFFFFF
#define TPM_ERROR_BASE       0x80000000

/* Forward declarations */
struct tpm_config;
struct session_map;
struct pcr_bank;
struct nv_index_map;
struct key_hierarchy;
struct tpm_session_pool;
struct tpm_error_handler;

/* Type definitions */
typedef struct tpm_config tpm_config_t;
typedef struct session_map session_map_t;
typedef struct pcr_bank pcr_bank_t;
typedef struct nv_index_map nv_index_map_t;
typedef struct key_hierarchy key_hierarchy_t;
typedef struct tpm_session_pool tpm_session_pool_t;
typedef struct tpm_error_handler tpm_error_handler_t;
typedef uint32_t tpm_status_t;
typedef uint64_t key_handle_t;

/**
 * @brief TPM initialization flags
 */
typedef enum {
    TPM_INIT_DEFAULT = 0x00000000,
    TPM_INIT_FORCE_CLEAR = 0x00000001,
    TPM_INIT_STRICT_POLICY = 0x00000002,
    TPM_INIT_DEBUG_MODE = 0x00000004
} tpm_init_flags_t;

/**
 * @brief TPM Manager class for comprehensive TPM lifecycle and security management
 */
class TpmManager {
public:
    /**
     * @brief Get singleton instance of TPM Manager
     * @param config Pointer to TPM configuration
     * @param flags Initialization flags
     * @return TpmManager& Reference to singleton instance
     */
    static TpmManager& getInstance(tpm_config_t* config = nullptr, uint32_t flags = TPM_INIT_DEFAULT);

    /**
     * @brief Create primary key in TPM hierarchy
     * @param key_template Key template defining attributes
     * @param hierarchy Target hierarchy for key creation
     * @param auth Authorization value
     * @return key_handle_t Handle to created key
     */
    key_handle_t createPrimaryKey(
        const tpm2_template_t* key_template,
        ESYS_TR hierarchy,
        const TPM2B_AUTH* auth
    );

    /**
     * @brief Perform platform attestation
     * @param pcr_selection PCR selection for attestation
     * @param nonce Nonce for freshness
     * @return tpm_status_t Attestation status
     */
    tpm_status_t attestPlatform(
        const TPML_PCR_SELECTION* pcr_selection,
        const TPM2B_NONCE* nonce
    );

    /**
     * @brief Manage TPM sessions
     * @param operation Session operation type
     * @param params Operation parameters
     * @return tpm_status_t Operation status
     */
    tpm_status_t manageSessions(
        uint32_t operation,
        void* params
    );

private:
    ESYS_CONTEXT* esys_context;
    session_map_t* active_sessions;
    pcr_bank_t* pcr_banks;
    nv_index_map_t* nv_indices;
    key_hierarchy_t* key_hierarchy;
    uint32_t capability_flags;
    tpm_session_pool_t* session_pool;
    tpm_error_handler_t* error_handler;

    TpmManager(tpm_config_t* config, uint32_t flags);
    ~TpmManager();

    /* Prevent copying */
    TpmManager(const TpmManager&) = delete;
    TpmManager& operator=(const TpmManager&) = delete;
};

/**
 * @brief Initialize TPM device and establish ESYS context
 * @param flags Initialization flags
 * @param config TPM configuration
 * @return tpm_status_t Initialization status
 */
tpm_status_t tpm_initialize(uint32_t flags, tpm_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* TALD_TPM_INTERFACE_H */