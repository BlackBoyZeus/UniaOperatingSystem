/*
 * TALD UNIA Memory Protection Subsystem
 * Version: 1.0.0
 *
 * Enhanced kernel-level memory protection with ASLR (40-bit entropy),
 * DEP, secure memory management, and real-time integrity monitoring
 * for game state, LiDAR data, and fleet communication buffers.
 */

#ifndef _MEMORY_PROTECTION_H_
#define _MEMORY_PROTECTION_H_

/* FreeBSD System Headers - v9.0 */
#include <sys/types.h>
#include <sys/mman.h>
#include <machine/param.h>

/* Internal Headers */
#include "tald_core.h"

/* Version Information */
#define MEMORY_PROTECTION_VERSION "1.0.0"

/* Memory Protection Constants */
#define MEMORY_PAGE_SIZE         4096
#define MAX_SECURE_REGIONS       32
#define ASLR_ENTROPY_BITS       40
#define GUARD_PAGE_SIZE         4096
#define MAX_FLEET_BUFFERS       32
#define ENTROPY_UPDATE_INTERVAL  3600    /* Entropy refresh interval in seconds */
#define INTEGRITY_CHECK_INTERVAL 100     /* Integrity check interval in ms */

/* Protection Level Enumeration */
enum protection_level {
    PROTECTION_NONE = 0,
    PROTECTION_READ_ONLY = 1,
    PROTECTION_READ_WRITE = 2,
    PROTECTION_EXECUTE = 3,
    PROTECTION_SECURE_DMA = 4
};

/* Memory Region Structure */
struct memory_region {
    void* base_address;
    size_t size;
    uint8_t protection_level;
    bool encrypted;
    void* guard_pages[2];
    uint32_t checksum;
    struct timespec last_check;
    uint32_t access_count;
    uint32_t violation_count;
} __packed;

/* Memory Protection Context */
struct memory_protection_ctx {
    uint32_t total_regions;
    struct memory_region regions[MAX_SECURE_REGIONS];
    uint64_t aslr_seed;
    uint32_t entropy_update_counter;
    struct timespec last_entropy_update;
    uint32_t integrity_violations;
    uint32_t thermal_violations;
    struct mtx region_lock;
} __packed;

/* DMA Protection Structure */
struct dma_protection {
    void* buffer;
    size_t size;
    uint32_t flags;
    bool locked;
    uint32_t access_mask;
    struct timespec last_access;
} __packed;

/**
 * Enhanced Memory Protection Management Class
 */
class MemoryProtector {
public:
    /**
     * Initialize memory protection system
     * @param config Core configuration
     * @param thermal_ctx Thermal monitoring context
     * @throws std::runtime_error on initialization failure
     */
    MemoryProtector(struct tald_core_config* config,
                   struct thermal_context* thermal_ctx);

    /**
     * Apply protection to memory region
     * @param region Memory region pointer
     * @param size Region size
     * @param flags Protection flags
     * @param enable_encryption Enable memory encryption
     * @return 0 on success, error code on failure
     */
    [[nodiscard]]
    int protect_region(void* region,
                      size_t size,
                      uint8_t flags,
                      bool enable_encryption);

    /**
     * Monitor region integrity
     * @param region_id Region identifier
     * @return 0 if integrity verified, error code on violation
     */
    [[nodiscard]]
    int monitor_integrity(uint32_t region_id);

private:
    struct memory_protection_ctx ctx;
    struct thermal_context* thermal;
    struct dma_protection dma_regions[MAX_SECURE_REGIONS];
    
    [[nodiscard]]
    int setup_aslr();
    
    [[nodiscard]]
    int configure_dep();
    
    [[nodiscard]]
    int init_integrity_monitoring();
    
    [[nodiscard]]
    int setup_dma_protection();

    // Prevent copying
    MemoryProtector(const MemoryProtector&) = delete;
    MemoryProtector& operator=(const MemoryProtector&) = delete;
};

/**
 * Initialize enhanced memory protection subsystem
 * @param config Core configuration
 * @param thermal_ctx Thermal monitoring context
 * @return 0 on success, error code on failure
 */
[[nodiscard]] __init __must_check
int init_memory_protection(struct tald_core_config* config,
                         struct thermal_context* thermal_ctx);

/**
 * Allocate memory with enhanced protection
 * @param size Memory size
 * @param protection_level Protection level
 * @param use_encryption Enable memory encryption
 * @return Protected memory pointer or NULL on failure
 */
[[nodiscard]] __must_check __cache_aligned
void* secure_alloc(size_t size,
                  uint8_t protection_level,
                  bool use_encryption);

/* Error Codes */
#define MEMORY_PROTECTION_SUCCESS      0
#define MEMORY_PROTECTION_ERROR_INIT  -1
#define MEMORY_PROTECTION_ERROR_PARAM -2
#define MEMORY_PROTECTION_ERROR_MEM   -3
#define MEMORY_PROTECTION_ERROR_PROT  -4
#define MEMORY_PROTECTION_ERROR_DMA   -5
#define MEMORY_PROTECTION_ERROR_THERM -6
#define MEMORY_PROTECTION_ERROR_INT   -7

/* Protection Flags */
#define MEMORY_PROT_ASLR      (1 << 0)
#define MEMORY_PROT_DEP       (1 << 1)
#define MEMORY_PROT_GUARD     (1 << 2)
#define MEMORY_PROT_ENCRYPT   (1 << 3)
#define MEMORY_PROT_DMA       (1 << 4)
#define MEMORY_PROT_THERMAL   (1 << 5)
#define MEMORY_PROT_INTEGRITY (1 << 6)

/* Kernel Attributes */
#define __kernel_export     __attribute__((visibility("default")))
#define __kernel_packed    __attribute__((packed))
#define __cache_aligned    __attribute__((aligned(64)))

/* Export symbols for kernel module use */
__kernel_export extern const struct memory_protection_ctx* get_protection_context(void);
__kernel_export extern int get_region_status(uint32_t region_id);
__kernel_export extern int set_protection_level(uint32_t region_id, uint8_t level);

#endif /* _MEMORY_PROTECTION_H_ */