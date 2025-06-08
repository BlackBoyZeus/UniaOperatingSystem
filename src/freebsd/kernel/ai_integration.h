/*
 * UNIA Operating System
 * AI Integration with FreeBSD Kernel - Header File
 */

#ifndef _UNIA_AI_INTEGRATION_H_
#define _UNIA_AI_INTEGRATION_H_

#include <sys/types.h>
#include <sys/malloc.h>
#include <vm/vm_object.h>

/* Version information */
#define UNIA_AI_VERSION_MAJOR 1
#define UNIA_AI_VERSION_MINOR 0
#define UNIA_AI_VERSION_PATCH 0
#define UNIA_AI_VERSION ((UNIA_AI_VERSION_MAJOR << 16) | \
                        (UNIA_AI_VERSION_MINOR << 8) | \
                        UNIA_AI_VERSION_PATCH)

/* Feature flags */
#define UNIA_AI_FEATURE_INFERENCE    0x00000001
#define UNIA_AI_FEATURE_GAME_AI      0x00000002
#define UNIA_AI_FEATURE_DISTRIBUTED  0x00000004
#define UNIA_AI_FEATURE_TENSORRT     0x00000008
#define UNIA_AI_FEATURE_VULKAN       0x00000010

/* Memory allocation flags */
#define UNIA_AI_MEM_DEVICE           0x00000001  /* Device memory (GPU) */
#define UNIA_AI_MEM_HOST             0x00000002  /* Host memory (CPU) */
#define UNIA_AI_MEM_SHARED           0x00000004  /* Shared between CPU and GPU */
#define UNIA_AI_MEM_CACHED           0x00000008  /* Cached memory */
#define UNIA_AI_MEM_UNCACHED         0x00000010  /* Uncached memory */
#define UNIA_AI_MEM_WRITE_COMBINED   0x00000020  /* Write-combined memory */

/* Task types */
#define UNIA_AI_TASK_INFERENCE       0x00000001  /* Run inference */
#define UNIA_AI_TASK_LOAD_MODEL      0x00000002  /* Load AI model */
#define UNIA_AI_TASK_UNLOAD_MODEL    0x00000003  /* Unload AI model */
#define UNIA_AI_TASK_NPC_BEHAVIOR    0x00000004  /* NPC behavior computation */
#define UNIA_AI_TASK_PROCEDURAL_GEN  0x00000005  /* Procedural content generation */
#define UNIA_AI_TASK_PLAYER_MODEL    0x00000006  /* Player modeling */

/* Task status */
#define UNIA_AI_TASK_STATUS_QUEUED       0x00000001  /* Task is queued */
#define UNIA_AI_TASK_STATUS_RUNNING      0x00000002  /* Task is running */
#define UNIA_AI_TASK_STATUS_COMPLETED    0x00000003  /* Task completed successfully */
#define UNIA_AI_TASK_STATUS_FAILED       0x00000004  /* Task failed */

/* Memory allocator */
MALLOC_DECLARE(M_UNIA_AI);

/* Structures */

/* AI module information */
struct unia_ai_info {
    uint32_t version;           /* Version number */
    uint32_t features;          /* Supported features */
    uint32_t max_models;        /* Maximum number of loaded models */
    uint32_t max_concurrent_inferences; /* Maximum concurrent inferences */
    uint32_t reserved_memory_mb; /* Reserved memory in MB */
};

/* Shared memory handle */
struct unia_ai_shm_handle {
    uint64_t id;                /* Unique identifier */
    size_t size;                /* Size in bytes */
    uint32_t flags;             /* Memory flags */
};

/* Shared memory region */
struct unia_ai_shm_region {
    uint64_t id;                /* Unique identifier */
    void *addr;                 /* Kernel virtual address */
    size_t size;                /* Size in bytes */
    vm_object_t object;         /* VM object */
};

/* AI task */
struct unia_ai_task {
    uint64_t id;                /* Unique identifier */
    uint32_t type;              /* Task type */
    uint32_t flags;             /* Task flags */
    uint32_t status;            /* Task status */
    uint32_t error_code;        /* Error code (if failed) */
    struct unia_ai_shm_handle input_handle;  /* Input data */
    struct unia_ai_shm_handle output_handle; /* Output data */
    struct cv completion_cv;    /* Completion condition variable */
    TAILQ_ENTRY(unia_ai_task) entries; /* Queue entries */
};

/* AI task result */
struct unia_ai_task_result {
    uint32_t status;            /* Task status */
    uint32_t error_code;        /* Error code (if failed) */
    struct unia_ai_shm_handle output_handle; /* Output data */
};

/* Function prototypes */

/* Initialize the AI integration module */
int unia_ai_init(void);

/* Cleanup the AI integration module */
void unia_ai_cleanup(void);

/* Get information about the AI integration module */
int unia_ai_get_info(struct unia_ai_info *info);

/* Allocate shared memory for AI operations */
int unia_ai_alloc_shared_memory(size_t size, uint32_t flags, 
                               struct unia_ai_shm_handle *handle);

/* Free shared memory */
int unia_ai_free_shared_memory(struct unia_ai_shm_handle handle);

/* Map shared memory into user space */
int unia_ai_map_shared_memory(struct thread *td, struct unia_ai_shm_handle handle,
                             void **addr);

/* Unmap shared memory from user space */
int unia_ai_unmap_shared_memory(struct thread *td, void *addr, size_t size);

/* Submit an AI task for processing */
int unia_ai_submit_task(struct unia_ai_task *task);

/* Wait for an AI task to complete */
int unia_ai_wait_task(uint64_t task_id, int timeout_ms);

/* Get the result of a completed AI task */
int unia_ai_get_task_result(uint64_t task_id, struct unia_ai_task_result *result);

/* Set real-time priority for AI tasks */
int unia_ai_set_rt_priority(int enable);

/* Reserve memory for AI operations */
int unia_ai_reserve_memory(size_t size_mb);

#endif /* _UNIA_AI_INTEGRATION_H_ */
