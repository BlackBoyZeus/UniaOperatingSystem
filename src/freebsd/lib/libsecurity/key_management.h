/**
 * @file key_management.h
 * @brief Hardware-backed key management interface for TALD UNIA security subsystem
 * @version 1.0
 * 
 * Provides comprehensive key lifecycle management with TPM 2.0 integration,
 * automated rotation, and secure deletion capabilities.
 */

#ifndef TALD_KEY_MANAGEMENT_H
#define TALD_KEY_MANAGEMENT_H

#include <openssl/crypto.h>    // OpenSSL 3.0.0
#include <tss2/tss2_esys.h>    // TPM2-TSS 3.2.0
#include <syslog.h>
#include "../config/security.conf"

#ifdef __cplusplus
extern "C" {
#endif

/* Constants */
#define KEY_SIZE_BITS 256
#define MAX_KEY_LIFETIME_DAYS 90
#define KEY_ROTATION_THRESHOLD_DAYS 80
#define MAX_ACTIVE_KEYS 32
#define TPM_CONTEXT_SIZE 1024
#define KEY_ROTATION_BATCH_SIZE 5
#define SECURE_DELETION_PASSES 3

/* Supported TPM manufacturers */
static const char* TPM_MANUFACTURER_WHITELIST[] = {"Intel", "AMD", "Infineon"};
#define TPM_FIRMWARE_MIN_VERSION "2.0.0"

/* Type definitions */
typedef uint32_t key_handle_t;

typedef enum {
    KEY_TYPE_AES_256_GCM,
    KEY_TYPE_RSA_4096,
    KEY_TYPE_ECC_P384,
    KEY_TYPE_HMAC_SHA384
} key_type_t;

typedef struct {
    uint32_t lifetime_days;
    bool hardware_backed;
    bool exportable;
    bool requires_authorization;
    char* usage_domain;
    uint32_t max_operations;
} key_attributes_t;

typedef struct {
    char manufacturer[64];
    char firmware_version[32];
    uint32_t capabilities;
    bool attestation_supported;
} tpm_manufacturer_info_t;

typedef struct {
    uint32_t rotation_interval_days;
    bool preserve_old_key;
    uint32_t migration_batch_size;
    void (*migration_callback)(key_handle_t old_key, key_handle_t new_key);
} rotation_config_t;

typedef struct {
    key_type_t type;
    key_attributes_t attributes;
    char* policy_digest;
    uint32_t auth_policy_size;
} enhanced_key_spec_t;

typedef struct {
    bool require_platform_auth;
    bool require_admin_auth;
    uint32_t min_auth_strength;
    char* allowed_operations;
} key_security_policy_t;

/* Function declarations */
/**
 * @brief Generates a new hardware-backed cryptographic key
 * @param key_type Type of key to generate
 * @param key_attributes Key attributes and lifecycle parameters
 * @param tpm_info TPM manufacturer information for validation
 * @return Handle to generated key or error code
 */
__attribute__((hardware_backed)) __attribute__((audit_log))
key_handle_t generate_key(key_type_t key_type, 
                         key_attributes_t* key_attributes,
                         tpm_manufacturer_info_t* tpm_info);

/**
 * @brief Performs secure key rotation with backup procedures
 * @param key_handle Handle to key requiring rotation
 * @param rotation_config Rotation parameters and callbacks
 * @return Handle to new rotated key or error code
 */
__attribute__((hardware_backed)) __attribute__((audit_log)) __attribute__((transactional))
key_handle_t rotate_key(key_handle_t key_handle, rotation_config_t* rotation_config);

/* KeyManager class definition */
typedef struct KeyManager KeyManager;

/**
 * @brief Creates a singleton instance of the KeyManager
 * @param config Key manager configuration parameters
 * @return Pointer to KeyManager instance or NULL on error
 */
KeyManager* KeyManager_getInstance(key_manager_config_t* config);

/**
 * @brief Creates a new managed key with enhanced security
 * @param manager KeyManager instance
 * @param key_spec Key specification and parameters
 * @param security_policy Security policy for key usage
 * @return Handle to created key or error code
 */
key_handle_t KeyManager_createKey(KeyManager* manager,
                                enhanced_key_spec_t* key_spec,
                                key_security_policy_t* security_policy);

/**
 * @brief Rotates a managed key with security controls
 * @param manager KeyManager instance
 * @param key_handle Handle to key requiring rotation
 * @param rotation_config Rotation parameters
 * @return Handle to new key or error code
 */
key_handle_t KeyManager_rotateKey(KeyManager* manager,
                                key_handle_t key_handle,
                                rotation_config_t* rotation_config);

/**
 * @brief Validates TPM manufacturer and firmware
 * @param manager KeyManager instance
 * @param tpm_info TPM manufacturer information
 * @return true if TPM is valid, false otherwise
 */
bool KeyManager_validateTPM(KeyManager* manager,
                          tpm_manufacturer_info_t* tpm_info);

/* Error codes */
typedef enum {
    KEY_SUCCESS = 0,
    KEY_ERROR_INVALID_PARAMS = -1,
    KEY_ERROR_TPM_VALIDATION = -2,
    KEY_ERROR_HARDWARE_FAILURE = -3,
    KEY_ERROR_ROTATION_FAILED = -4,
    KEY_ERROR_POLICY_VIOLATION = -5,
    KEY_ERROR_BACKUP_FAILED = -6,
    KEY_ERROR_QUOTA_EXCEEDED = -7
} key_error_t;

#ifdef __cplusplus
}
#endif

#endif /* TALD_KEY_MANAGEMENT_H */