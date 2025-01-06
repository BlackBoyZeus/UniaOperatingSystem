/**
 * @file secure_storage.c
 * @brief Implementation of hardware-backed secure storage for TALD UNIA
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <syslog.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "secure_storage.h"
#include "key_management.h"
#include "tpm_interface.h"

/* OpenSSL 3.0.0 */
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/err.h>

/* TPM2-TSS 3.2.0 */
#include <tss2/tss2_esys.h>

/* Static assertions */
static_assert(STORAGE_VERSION == 1, "Invalid storage version");
static_assert(SECURE_DELETE_PASSES >= 3, "Insufficient secure deletion passes");

/* Internal structures */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t size;
    uint64_t created_time;
    uint64_t last_access;
    uint8_t iv[16];
    uint8_t tag[16];
    uint8_t reserved[32];
} storage_header_t;

typedef struct {
    uint8_t* data;
    size_t size;
    uint8_t iv[16];
    uint8_t tag[16];
} encrypted_block_t;

/* Singleton instance */
static SecureStorageImpl* instance = NULL;

/* Implementation of SecureStorageImpl */
class SecureStorageImpl {
private:
    KeyManager* key_manager;
    TpmManager* tpm_manager;
    storage_map_t* storage_map;
    key_set_t* active_keys;
    QuotaManager* quota_manager;
    AuditLogger* audit_logger;

    /* Private constructor */
    SecureStorageImpl() {
        /* Initialize components */
        key_manager = KeyManager_getInstance(NULL);
        tpm_manager = &TpmManager::getInstance();
        storage_map = calloc(1, sizeof(storage_map_t));
        active_keys = calloc(1, sizeof(key_set_t));
        quota_manager = new QuotaManager();
        audit_logger = new AuditLogger();

        /* Configure key rotation */
        rotation_config_t rot_config = {
            .rotation_interval_days = 7,
            .preserve_old_key = true,
            .migration_batch_size = KEY_ROTATION_BATCH_SIZE,
            .migration_callback = handle_key_migration
        };
        key_manager->schedule_rotation(&rot_config);
    }

public:
    static SecureStorageImpl* getInstance() {
        if (!instance) {
            instance = new SecureStorageImpl();
        }
        return instance;
    }

    status_t rotate_storage_keys() {
        audit_logger->log_event(AUDIT_KEY_ROTATION_START);
        
        /* Validate TPM manufacturer */
        tpm_manufacturer_info_t tpm_info;
        if (!tpm_manager->validate_manufacturer(&tpm_info)) {
            audit_logger->log_event(AUDIT_TPM_VALIDATION_FAILED);
            return STORAGE_ERROR_TPM_VALIDATION;
        }

        /* Create new primary key */
        key_handle_t new_key = tpm_manager->create_primary_key();
        if (!new_key) {
            audit_logger->log_event(AUDIT_KEY_CREATION_FAILED);
            return STORAGE_ERROR_HARDWARE;
        }

        /* Re-encrypt all storage blocks */
        status_t status = reencrypt_storage_blocks(new_key);
        if (status != STORAGE_SUCCESS) {
            audit_logger->log_event(AUDIT_REENCRYPTION_FAILED);
            return status;
        }

        audit_logger->log_event(AUDIT_KEY_ROTATION_COMPLETE);
        return STORAGE_SUCCESS;
    }

private:
    status_t reencrypt_storage_blocks(key_handle_t new_key) {
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return STORAGE_ERROR_ENCRYPTION;

        /* Process each storage block */
        for (size_t i = 0; i < storage_map->count; i++) {
            storage_block_t* block = &storage_map->blocks[i];
            
            /* Decrypt with old key */
            encrypted_block_t dec_block;
            status_t status = decrypt_block(block, &dec_block);
            if (status != STORAGE_SUCCESS) return status;

            /* Re-encrypt with new key */
            status = encrypt_block(&dec_block, new_key, block);
            secure_zero(&dec_block, sizeof(dec_block));
            if (status != STORAGE_SUCCESS) return status;
        }

        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_SUCCESS;
    }

    static void handle_key_migration(key_handle_t old_key, key_handle_t new_key) {
        audit_logger->log_event(AUDIT_KEY_MIGRATION, old_key, new_key);
    }
};

/* Implementation of secure_store */
__attribute__((hardware_backed)) __attribute__((audit_logged))
status_t secure_store_impl(const uint8_t* data, size_t size, storage_id_t storage_id,
                          integrity_token_t* integrity_token) {
    if (!data || !size || size > MAX_STORAGE_SIZE || !integrity_token) {
        return STORAGE_ERROR_INVALID_PARAMS;
    }

    SecureStorageImpl* storage = SecureStorageImpl::getInstance();
    
    /* Validate TPM manufacturer */
    tpm_manufacturer_info_t tpm_info;
    if (!storage->tpm_manager->validate_manufacturer(&tpm_info)) {
        return STORAGE_ERROR_TPM_VALIDATION;
    }

    /* Check storage quota */
    if (!storage->quota_manager->check_quota(storage_id, size)) {
        return STORAGE_ERROR_QUOTA;
    }

    /* Initialize encryption context */
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return STORAGE_ERROR_ENCRYPTION;

    /* Generate random IV using TPM */
    uint8_t iv[16];
    if (!RAND_bytes(iv, sizeof(iv))) {
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    /* Encrypt data in blocks */
    encrypted_block_t enc_block;
    status_t status = encrypt_data(data, size, iv, &enc_block);
    if (status != STORAGE_SUCCESS) {
        EVP_CIPHER_CTX_free(ctx);
        return status;
    }

    /* Calculate integrity token */
    status = calculate_integrity_token(&enc_block, integrity_token);
    if (status != STORAGE_SUCCESS) {
        secure_zero(&enc_block, sizeof(enc_block));
        EVP_CIPHER_CTX_free(ctx);
        return status;
    }

    /* Write to storage */
    status = write_encrypted_block(storage_id, &enc_block);
    secure_zero(&enc_block, sizeof(enc_block));
    EVP_CIPHER_CTX_free(ctx);

    storage->audit_logger->log_event(AUDIT_STORE_COMPLETE, storage_id);
    return status;
}

/* Implementation of secure deletion */
__attribute__((hardware_backed)) __attribute__((audit_logged))
status_t secure_delete_impl(storage_id_t storage_id) {
    SecureStorageImpl* storage = SecureStorageImpl::getInstance();
    
    /* Validate storage ID */
    if (!storage->storage_map->validate_id(storage_id)) {
        return STORAGE_ERROR_INVALID_PARAMS;
    }

    /* Perform secure deletion passes */
    for (int pass = 0; pass < SECURE_DELETE_PASSES; pass++) {
        uint8_t pattern[STORAGE_BLOCK_SIZE];
        
        /* Generate random pattern for each pass */
        if (!RAND_bytes(pattern, sizeof(pattern))) {
            return STORAGE_ERROR_HARDWARE;
        }

        /* Overwrite storage blocks */
        status_t status = overwrite_storage(storage_id, pattern, sizeof(pattern));
        if (status != STORAGE_SUCCESS) {
            return status;
        }

        /* Verify overwrite */
        status = verify_overwrite(storage_id, pattern, sizeof(pattern));
        if (status != STORAGE_SUCCESS) {
            return status;
        }
    }

    /* Release storage key from TPM */
    key_handle_t key_handle = storage->storage_map->get_key_handle(storage_id);
    if (key_handle) {
        storage->tpm_manager->release_key(key_handle);
    }

    /* Update storage map */
    storage->storage_map->remove_storage(storage_id);
    storage->audit_logger->log_event(AUDIT_DELETE_COMPLETE, storage_id);

    return STORAGE_SUCCESS;
}

/* Helper functions */
static void secure_zero(void* ptr, size_t size) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    while (size--) *p++ = 0;
}

static status_t encrypt_data(const uint8_t* data, size_t size, const uint8_t* iv,
                           encrypted_block_t* out_block) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return STORAGE_ERROR_ENCRYPTION;

    /* Initialize encryption */
    if (!EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    /* Set IV length and IV */
    if (!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL) ||
        !EVP_EncryptInit_ex(ctx, NULL, NULL, key_manager->get_active_key(), iv)) {
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    /* Encrypt data */
    int outlen;
    out_block->data = (uint8_t*)malloc(size + EVP_MAX_BLOCK_LENGTH);
    if (!EVP_EncryptUpdate(ctx, out_block->data, &outlen, data, size)) {
        free(out_block->data);
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    /* Finalize encryption */
    int final_len;
    if (!EVP_EncryptFinal_ex(ctx, out_block->data + outlen, &final_len)) {
        free(out_block->data);
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    /* Get tag */
    if (!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, out_block->tag)) {
        free(out_block->data);
        EVP_CIPHER_CTX_free(ctx);
        return STORAGE_ERROR_ENCRYPTION;
    }

    out_block->size = outlen + final_len;
    memcpy(out_block->iv, iv, 16);
    
    EVP_CIPHER_CTX_free(ctx);
    return STORAGE_SUCCESS;
}

/* Export symbols */
extern "C" {
    SecureStorageImpl* get_secure_storage_instance() {
        return SecureStorageImpl::getInstance();
    }

    status_t secure_store(const uint8_t* data, size_t size, storage_id_t storage_id,
                         integrity_token_t* integrity_token) {
        return secure_store_impl(data, size, storage_id, integrity_token);
    }

    status_t secure_delete(storage_id_t storage_id) {
        return secure_delete_impl(storage_id);
    }
}