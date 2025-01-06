/* TALD UNIA Second Stage Bootloader
 * Version: 1.0
 * GNU Assembler x86_32 v2.40
 */

/* Constants */
.set BOOT1_LOAD_ADDRESS,    0x7E00
.set KERNEL_LOAD_ADDRESS,   0x100000
.set TPM_EXTEND_COMMAND,    0xC1
.set LIDAR_BASE_ADDRESS,    0xFED50000
.set GPU_INIT_ADDRESS,      0xFED60000
.set TPM_PCR_BANK,         0x0
.set HARDWARE_SIG_OFFSET,   0x1000
.set ERROR_BUFFER,         0x7000
.set DEBUG_PORT,           0x3F8

/* GDT entries */
.section .data
gdt:
    .quad 0x0000000000000000    /* Null descriptor */
    .quad 0x00CF9A000000FFFF    /* Code descriptor */
    .quad 0x00CF92000000FFFF    /* Data descriptor */
gdt_end:

gdt_desc:
    .word gdt_end - gdt - 1     /* GDT size */
    .long gdt                    /* GDT address */

/* Error messages in rodata */
.section .rodata
err_tpm:     .asciz "TPM Error: "
err_lidar:   .asciz "LiDAR Error: "
err_gpu:     .asciz "GPU Error: "
err_kernel:  .asciz "Kernel Error: "

/* BSS section for uninitialized data */
.section .bss
.align 16
stack_bottom:
    .skip 4096
stack_top:

tpm_buffer:
    .skip 256
hw_state:
    .skip 512

/* Main code section */
.section .text
.code16

/* Entry point */
.global boot1_start
.type boot1_start, @function
boot1_start:
    /* Disable interrupts */
    cli
    
    /* Save boot drive number */
    movb %dl, boot_drive

    /* Enable A20 line */
    call enable_a20
    
    /* Load GDT and switch to protected mode */
    lgdt gdt_desc
    movl %cr0, %eax
    orl $1, %eax
    movl %eax, %cr0
    
    /* Jump to 32-bit code */
    ljmp $0x08, $protected_mode

.code32
protected_mode:
    /* Setup segment registers */
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %gs
    movw %ax, %ss
    
    /* Setup stack */
    movl $stack_top, %esp
    
    /* Initialize critical hardware */
    call init_critical_hw
    testl %eax, %eax
    jnz boot_error
    
    /* Extend TPM measurements */
    movl $hw_state, %esi
    movl $512, %ecx
    call extend_tpm_measurements
    testl %eax, %eax
    jnz tpm_error
    
    /* Load kernel */
    call load_kernel
    jc kernel_error
    
    /* Jump to kernel */
    jmp *KERNEL_LOAD_ADDRESS

/* Hardware initialization function */
.type init_critical_hw, @function
init_critical_hw:
    pushl %ebx
    
    /* Initialize LiDAR */
    movl $LIDAR_BASE_ADDRESS, %ebx
    movl $0x1, (%ebx)           /* Enable LiDAR */
    movl 4(%ebx), %eax          /* Read status */
    testl $0x1, %eax
    jz lidar_error
    
    /* Initialize GPU */
    movl $GPU_INIT_ADDRESS, %ebx
    movl $0x1, (%ebx)           /* Enable GPU */
    movl 4(%ebx), %eax          /* Read status */
    testl $0x1, %eax
    jz gpu_error
    
    xorl %eax, %eax             /* Return success */
    popl %ebx
    ret

/* TPM measurement function */
.type extend_tpm_measurements, @function
extend_tpm_measurements:
    pushl %ebx
    pushl %edx
    
    /* Prepare TPM command */
    movl $TPM_EXTEND_COMMAND, %ebx
    movl $TPM_PCR_BANK, %edx
    
    /* Execute TPM extend */
    movl $tpm_buffer, %edi
    movl %ebx, (%edi)
    movl %edx, 4(%edi)
    movl %esi, 8(%edi)
    movl %ecx, 12(%edi)
    
    /* Send command to TPM */
    /* TPM communication code here */
    
    popl %edx
    popl %ebx
    ret

/* Kernel loading function */
.type load_kernel, @function
load_kernel:
    pushl %ebx
    pushl %esi
    pushl %edi
    
    /* Load kernel from disk */
    movl $KERNEL_LOAD_ADDRESS, %edi
    /* Disk loading code here */
    
    /* Verify kernel signature */
    movl $HARDWARE_SIG_OFFSET, %esi
    /* Signature verification code here */
    
    popl %edi
    popl %esi
    popl %ebx
    ret

/* Error handlers */
boot_error:
    movl $err_tpm, %esi
    jmp error_handler

tpm_error:
    movl $err_tpm, %esi
    jmp error_handler

lidar_error:
    movl $err_lidar, %esi
    jmp error_handler

gpu_error:
    movl $err_gpu, %esi
    jmp error_handler

kernel_error:
    movl $err_kernel, %esi
    jmp error_handler

/* Generic error handler */
error_handler:
    /* Write error to debug port */
    movl $DEBUG_PORT, %edx
1:
    movb (%esi), %al
    testb %al, %al
    jz 2f
    outb %al, %dx
    incl %esi
    jmp 1b
2:
    /* Halt system */
    cli
    hlt
    jmp 2b

/* Additional helper functions */
enable_a20:
    /* A20 enable code using keyboard controller */
    inb $0x64, %al
    testb $0x2, %al
    jnz enable_a20
    movb $0xd1, %al
    outb %al, $0x64
    inb $0x64, %al
    testb $0x2, %al
    jnz enable_a20
    movb $0xdf, %al
    outb %al, $0x60
    ret

/* Boot drive storage */
boot_drive:
    .byte 0