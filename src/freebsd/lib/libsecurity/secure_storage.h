/**
 * @file secure_storage.h
 * @brief Hardware-backed secure storage interface for TALD UNIA's security subsystem
 * @version 1.0
 * 
 * Provides comprehensive secure storage operations with TPM 2.0 integration,
 * hardware-backed encryption, and secure deletion capabilities.
 */

#ifndef TALD_SECURE_STORAGE_H
#define TALD_SECURE_STORAGE_H

#include <stdint.h>
#include <stdbool.h>
#include <openssl/crypto.h>    // OpenSSL 3.0.0
#include <tss2/tss2_esys.h>    // TPM2-TSS 3.2.0
#include "key_management.h"
#include "tpm_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Constants */
#define MAX_STORAGE_SIZE (1024 * 1024 * 1024)  // 1GB maximum storage size
#define STORAGE_BLOCK_SIZE 4096                 // 4KB block size
#define MAX_STORAGE_KEYS 16                     // Maximum concurrent storage keys
#define STORAGE_VERSION 1                       // Current storage version
#define KEY_ROTATION_INTERVAL (7 * 24 * 60 * 60) // 7 days key rotation interval
#define SECURE_DELETE_PASSES 3                  // Number of secure deletion passes

/* Type definitions */
typedef uint64_t storage_id_t;
typedef uint64_t integrity_token_t;

typedef enum {
    STORAGE_SUCCESS = 0,
    STORAGE_ERROR_INVALID_PARAMS = -1,
    STORAGE_ERROR_TPM_VALIDATION = -2,
    STORAGE_ERROR_ENCRYPTION = -3,
    STORAGE_ERROR_INTEGRITY = -4,
    STORAGE_ERROR_QUOTA = -5,
    STORAGE_ERROR_HARDWARE = -6,
    STORAGE_ERROR_ACCESS = -7,
    STORAGE_ERROR_VERSION = -8
} status_t;

typedef struct {
    uint32_t version;
    uint64_t max_size;
    bool require_encryption;
    bool require_integrity;
    bool enable_backup;
    char* access_domain;
} storage_config_t;

typedef struct {
    uint64_t current_usage;
    uint64_t max_usage;
    uint32_t operation_count;
    uint32_t max_operations;
} quota_params_t;

typedef struct {
    uint32_t passes;
    bool verify_deletion;
    bool preserve_metadata;
    void (*deletion_callback)(storage_id_t);
} secure_delete_config_t;

/**
 * @brief Secure storage management class with TPM integration
 */
class SecureStorage {
public:
    /**
     * @brief Get singleton instance of SecureStorage
     * @return Reference to SecureStorage instance
     */
    static SecureStorage& getInstance();

    /**
     * @brief Create new secure storage area
     * @param config Storage configuration parameters
     * @param quota_params Quota and usage parameters
     * @return storage_id_t ID of created storage
     */
    __attribute__((hardware_backed))
    storage_id_t create_storage(
        storage_config_t* config,
        quota_params_t* quota_params
    );

    /**
     * @brief Securely delete storage with multiple passes
     * @param storage_id Storage ID to delete
     * @param config Secure deletion configuration
     * @return status_t Operation status
     */
    __attribute__((hardware_backed))
    status_t delete_storage(
        storage_id_t storage_id,
        secure_delete_config_t* config
    );

    /**
     * @brief Rotate storage encryption keys
     * @param storage_id Storage ID for key rotation
     * @return status_t Operation status
     */
    __attribute__((hardware_backed))
    status_t rotate_keys(storage_id_t storage_id);

private:
    KeyManager* key_manager;
    TpmManager* tpm_manager;
    void* storage_map;
    void* active_keys;
    void* quota_manager;
    void* access_log;

    SecureStorage();
    ~SecureStorage();
    
    /* Prevent copying */
    SecureStorage(const SecureStorage&) = delete;
    SecureStorage& operator=(const SecureStorage&) = delete;
};

/**
 * @brief Store data securely with hardware-backed encryption
 * @param data Data buffer to store
 * @param size Size of data
 * @param storage_id Target storage ID
 * @param integrity_token Output integrity token
 * @return status_t Operation status
 */
__attribute__((hardware_backed)) __attribute__((integrity_verified))
status_t secure_store(
    const uint8_t* data,
    size_t size,
    storage_id_t storage_id,
    integrity_token_t* integrity_token
);

/**
 * @brief Retrieve data with integrity verification
 * @param buffer Output buffer for data
 * @param size Size of data
 * @param storage_id Source storage ID
 * @param integrity_token Expected integrity token
 * @return status_t Operation status
 */
__attribute__((hardware_backed)) __attribute__((integrity_verified))
status_t secure_retrieve(
    uint8_t* buffer,
    size_t* size,
    storage_id_t storage_id,
    const integrity_token_t* integrity_token
);

#ifdef __cplusplus
}
#endif

#endif /* TALD_SECURE_STORAGE_H */