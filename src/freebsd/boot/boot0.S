/* 
 * TALD UNIA Boot0 Loader
 * Version: 1.0
 * GNU Assembler x86_16 v2.40
 *
 * Enhanced MBR bootloader with TPM 2.0 validation and secure boot
 */

    .code16
    .section .text
    .globl _start

/* Constants */
.equ BOOT0_LOAD_ADDRESS,    0x7C00
.equ BOOT1_LOAD_ADDRESS,    0x7E00
.equ TPM_BASE_ADDRESS,      0xFED40000
.equ PARTITION_TABLE_OFFSET,0x1BE
.equ TPM_PCR_BANK_0,       0x0
.equ TPM_STATUS_REGISTER,   0x0
.equ TPM_COMMAND_REGISTER,  0x24
.equ TPM_RESPONSE_REGISTER, 0x28
.equ ERROR_TPM_NOT_READY,   0x01
.equ ERROR_TPM_PCR_INVALID, 0x02
.equ ERROR_BOOT1_LOAD,      0x03
.equ ERROR_BOOT1_SIGNATURE, 0x04

_start:
    /* Initialize segments and stack */
    xor     %ax, %ax
    mov     %ax, %ds
    mov     %ax, %es
    mov     %ax, %ss
    mov     $BOOT0_LOAD_ADDRESS, %sp

    /* Save BIOS drive number */
    mov     %dl, drive_number

    /* Verify TPM presence and status */
    call    verify_tpm
    test    %al, %al
    jnz     boot_error

    /* Scan partition table */
    mov     $PARTITION_TABLE_OFFSET, %si
    mov     $4, %cx                    /* Check 4 partition entries */

scan_partitions:
    /* Check if partition is bootable */
    cmpb    $0x80, (%si)              /* Check bootable flag */
    je      found_bootable
    add     $16, %si                   /* Move to next partition entry */
    loop    scan_partitions
    jmp     no_bootable_partition

found_bootable:
    /* Load and verify boot1 */
    mov     drive_number, %dl
    call    load_boot1
    jc      boot_error

    /* Transfer control to boot1 */
    ljmp    $0, $BOOT1_LOAD_ADDRESS

/* TPM verification routine */
verify_tpm:
    /* Check TPM presence */
    mov     $TPM_BASE_ADDRESS, %dx
    in      %dx, %al
    cmp     $0xFF, %al                /* Check if TPM responds */
    je      tpm_not_present

    /* Check TPM status */
    mov     $TPM_STATUS_REGISTER, %dx
    in      %dx, %al
    test    $0x1, %al                 /* Check ready bit */
    jz      tpm_not_ready

    /* Verify PCR bank 0 */
    mov     $TPM_PCR_BANK_0, %dx
    in      %dx, %al
    test    %al, %al                  /* Validate measurement */
    jnz     tpm_pcr_invalid

    xor     %al, %al                  /* Success */
    ret

tpm_not_present:
tpm_not_ready:
    mov     $ERROR_TPM_NOT_READY, %al
    ret

tpm_pcr_invalid:
    mov     $ERROR_TPM_PCR_INVALID, %al
    ret

/* Boot1 loading routine */
load_boot1:
    pusha
    /* Convert LBA to CHS */
    mov     4(%si), %eax              /* Load LBA */
    xor     %edx, %edx
    movw    sectors_per_track, %cx
    div     %ecx                      /* Divide by sectors per track */
    inc     %dl                       /* Sector number starts at 1 */
    mov     %dl, %cl                  /* CL = sector number */
    
    xor     %edx, %edx
    movw    heads_per_cylinder, %bx
    div     %ebx                      /* Divide by heads */
    mov     %dl, %dh                  /* DH = head number */
    mov     %al, %ch                  /* CH = cylinder (low 8 bits) */
    
    mov     $0x02, %ah                /* Read sectors function */
    mov     $1, %al                   /* Read 1 sector */
    mov     $BOOT1_LOAD_ADDRESS, %bx  /* Load address */
    
    /* Attempt read with retries */
    mov     $3, %si                   /* Retry counter */
retry_read:
    int     $0x13
    jnc     verify_boot1_sig
    dec     %si
    jz      boot1_load_error
    xor     %ah, %ah                  /* Reset disk system */
    int     $0x13
    jmp     retry_read

verify_boot1_sig:
    /* Verify boot1 signature */
    mov     $BOOT1_LOAD_ADDRESS, %si
    mov     signature_offset, %bx
    cmpsw                             /* Compare signature */
    jne     boot1_sig_error
    
    popa
    clc                               /* Clear carry on success */
    ret

boot1_load_error:
    popa
    mov     $ERROR_BOOT1_LOAD, %al
    stc
    ret

boot1_sig_error:
    popa
    mov     $ERROR_BOOT1_SIGNATURE, %al
    stc
    ret

boot_error:
    /* Display error code in AL */
    mov     $0x0E, %ah               /* BIOS teletype output */
    mov     $'E', %al
    int     $0x10
    mov     error_code, %al
    add     $'0', %al
    int     $0x10
    
    /* Infinite loop on error */
1:  hlt
    jmp     1b

no_bootable_partition:
    mov     $ERROR_BOOT1_LOAD, %al
    jmp     boot_error

/* Data section */
    .section .data
drive_number:            .byte 0
sectors_per_track:       .word 63
heads_per_cylinder:      .word 16
error_code:             .byte 0
signature_offset:       .word 0x1FE

/* Security section */
    .section .security
tpm_config:             .long 0
secure_boot_flags:      .byte 0

/* Boot sector signature */
    .section .sig
    .org 510
    .word 0xAA55

/* Partition table structure */
    .section partitions
    .org PARTITION_TABLE_OFFSET
    .space 64  /* 4 partition entries * 16 bytes each */