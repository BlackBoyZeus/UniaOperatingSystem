/**
 * @file key_management.c
 * @brief Implementation of hardware-backed key management system for TALD UNIA
 * @version 1.0
 */

#include "key_management.h"
#include "tpm_interface.h"
#include <openssl/crypto.h>    // OpenSSL 3.0.0
#include <tss2/tss2_esys.h>    // TPM2-TSS 3.2.0
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <syslog.h>

/* Static instance for singleton pattern */
static KeyManager* instance = NULL;
static pthread_mutex_t instance_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Internal structures */
typedef struct {
    key_handle_t handle;
    key_type_t type;
    time_t creation_time;
    time_t last_rotation;
    uint32_t operation_count;
    bool is_backed_up;
    encrypted_metadata_t metadata;
} key_record_t;

struct KeyManager {
    ESYS_CONTEXT* tpm_context;
    TpmManager* tpm_manager;
    key_record_t active_keys[MAX_ACTIVE_KEYS];
    uint32_t active_key_count;
    pthread_mutex_t key_mutex;
    pthread_t rotation_thread;
    bool shutdown_requested;
};

/* Internal helper functions */
static bool validate_tpm_manufacturer(const tpm_manufacturer_info_t* info) {
    if (!info) return false;
    
    for (size_t i = 0; i < sizeof(TPM_MANUFACTURER_WHITELIST) / sizeof(char*); i++) {
        if (strcmp(info->manufacturer, TPM_MANUFACTURER_WHITELIST[i]) == 0) {
            return true;
        }
    }
    return false;
}

static void secure_erase(void* data, size_t size) {
    volatile unsigned char* p = (volatile unsigned char*)data;
    for (int pass = 0; pass < SECURE_DELETION_PASSES; pass++) {
        for (size_t i = 0; i < size; i++) {
            p[i] = 0xFF;
            p[i] = 0x00;
            p[i] = 0xAA;
            p[i] = 0x55;
        }
    }
    memset((void*)data, 0, size);
}

/* Key rotation thread function */
static void* rotation_monitor(void* arg) {
    KeyManager* manager = (KeyManager*)arg;
    while (!manager->shutdown_requested) {
        pthread_mutex_lock(&manager->key_mutex);
        
        time_t current_time = time(NULL);
        for (uint32_t i = 0; i < manager->active_key_count; i++) {
            key_record_t* key = &manager->active_keys[i];
            double days_since_rotation = difftime(current_time, key->last_rotation) / (24 * 3600);
            
            if (days_since_rotation >= KEY_ROTATION_THRESHOLD_DAYS) {
                rotation_config_t config = {
                    .rotation_interval_days = MAX_KEY_LIFETIME_DAYS,
                    .preserve_old_key = true,
                    .migration_batch_size = KEY_ROTATION_BATCH_SIZE
                };
                KeyManager_rotateKey(manager, key->handle, &config);
            }
        }
        
        pthread_mutex_unlock(&manager->key_mutex);
        sleep(3600); // Check every hour
    }
    return NULL;
}

/* Implementation of public functions */
__attribute__((hardware_backed)) __attribute__((audit_log))
key_handle_t generate_key(key_type_t key_type, key_attributes_t* key_attributes,
                         tpm_manufacturer_info_t* tpm_info) {
    if (!key_attributes || !tpm_info) {
        syslog(LOG_ERR, "Invalid parameters in generate_key");
        return KEY_ERROR_INVALID_PARAMS;
    }

    // Validate TPM manufacturer
    if (!validate_tpm_manufacturer(tpm_info)) {
        syslog(LOG_ERR, "TPM manufacturer validation failed");
        return KEY_ERROR_TPM_VALIDATION;
    }

    KeyManager* manager = KeyManager_getInstance(NULL);
    if (!manager) {
        syslog(LOG_ERR, "Failed to get KeyManager instance");
        return KEY_ERROR_HARDWARE_FAILURE;
    }

    pthread_mutex_lock(&manager->key_mutex);

    // Check key quota
    if (manager->active_key_count >= MAX_ACTIVE_KEYS) {
        pthread_mutex_unlock(&manager->key_mutex);
        syslog(LOG_ERR, "Maximum active keys limit reached");
        return KEY_ERROR_QUOTA_EXCEEDED;
    }

    // Create TPM-backed key
    tpm2_template_t key_template = {
        .type = key_type,
        .size = KEY_SIZE_BITS,
        .attributes = key_attributes
    };

    key_handle_t handle = manager->tpm_manager->createPrimaryKey(
        &key_template,
        ESYS_TR_RH_OWNER,
        NULL
    );

    if (handle == 0) {
        pthread_mutex_unlock(&manager->key_mutex);
        syslog(LOG_ERR, "TPM key creation failed");
        return KEY_ERROR_HARDWARE_FAILURE;
    }

    // Initialize key record
    key_record_t* new_key = &manager->active_keys[manager->active_key_count];
    new_key->handle = handle;
    new_key->type = key_type;
    new_key->creation_time = time(NULL);
    new_key->last_rotation = new_key->creation_time;
    new_key->operation_count = 0;
    new_key->is_backed_up = false;

    manager->active_key_count++;

    pthread_mutex_unlock(&manager->key_mutex);

    // Log key generation
    syslog(LOG_NOTICE, "Generated new key: handle=%u, type=%d", handle, key_type);

    return handle;
}

KeyManager* KeyManager_getInstance(key_manager_config_t* config) {
    pthread_mutex_lock(&instance_mutex);
    
    if (!instance) {
        instance = (KeyManager*)calloc(1, sizeof(KeyManager));
        if (!instance) {
            pthread_mutex_unlock(&instance_mutex);
            return NULL;
        }

        // Initialize TPM context
        instance->tpm_manager = &TpmManager::getInstance();
        if (!instance->tpm_manager) {
            free(instance);
            instance = NULL;
            pthread_mutex_unlock(&instance_mutex);
            return NULL;
        }

        pthread_mutex_init(&instance->key_mutex, NULL);
        instance->shutdown_requested = false;

        // Start rotation monitor thread
        if (pthread_create(&instance->rotation_thread, NULL, rotation_monitor, instance) != 0) {
            pthread_mutex_destroy(&instance->key_mutex);
            free(instance);
            instance = NULL;
            pthread_mutex_unlock(&instance_mutex);
            return NULL;
        }
    }
    
    pthread_mutex_unlock(&instance_mutex);
    return instance;
}

key_handle_t KeyManager_rotateKey(KeyManager* manager, key_handle_t key_handle,
                                rotation_config_t* rotation_config) {
    if (!manager || !rotation_config) {
        return KEY_ERROR_INVALID_PARAMS;
    }

    pthread_mutex_lock(&manager->key_mutex);

    // Find key record
    key_record_t* key = NULL;
    for (uint32_t i = 0; i < manager->active_key_count; i++) {
        if (manager->active_keys[i].handle == key_handle) {
            key = &manager->active_keys[i];
            break;
        }
    }

    if (!key) {
        pthread_mutex_unlock(&manager->key_mutex);
        return KEY_ERROR_INVALID_PARAMS;
    }

    // Generate new key with same attributes
    key_attributes_t attributes = {
        .lifetime_days = rotation_config->rotation_interval_days,
        .hardware_backed = true,
        .exportable = false
    };

    tpm_manufacturer_info_t tpm_info;
    manager->tpm_manager->attestPlatform(NULL, NULL); // Verify TPM state
    
    key_handle_t new_handle = generate_key(key->type, &attributes, &tpm_info);
    if (new_handle <= 0) {
        pthread_mutex_unlock(&manager->key_mutex);
        return KEY_ERROR_ROTATION_FAILED;
    }

    // Update key record
    if (!rotation_config->preserve_old_key) {
        secure_erase(key, sizeof(key_record_t));
    }
    
    key->handle = new_handle;
    key->last_rotation = time(NULL);
    key->operation_count = 0;

    pthread_mutex_unlock(&manager->key_mutex);

    syslog(LOG_NOTICE, "Rotated key: old=%u, new=%u", key_handle, new_handle);
    
    return new_handle;
}

void KeyManager_destroy(KeyManager* manager) {
    if (!manager) return;

    manager->shutdown_requested = true;
    pthread_join(manager->rotation_thread, NULL);

    pthread_mutex_lock(&manager->key_mutex);
    
    // Securely erase all keys
    for (uint32_t i = 0; i < manager->active_key_count; i++) {
        secure_erase(&manager->active_keys[i], sizeof(key_record_t));
    }
    
    pthread_mutex_unlock(&manager->key_mutex);
    pthread_mutex_destroy(&manager->key_mutex);

    secure_erase(manager, sizeof(KeyManager));
    free(manager);

    pthread_mutex_lock(&instance_mutex);
    instance = NULL;
    pthread_mutex_unlock(&instance_mutex);
}