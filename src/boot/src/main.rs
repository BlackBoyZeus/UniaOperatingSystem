#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{allocator, hlt_loop, memory, println, serial_println};
use x86_64::VirtAddr;
use alloc::string::String;
use alloc::format;

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Direct VGA buffer manipulation for initial message
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        let message = b"UNIA OS Starting...";
        
        for (i, &byte) in message.iter().enumerate() {
            *vga_buffer.add(i * 2) = byte;
            *vga_buffer.add(i * 2 + 1) = 0x0F; // White on black
        }
    }
    
    serial_println!("STEP 1: UNIA OS Serial Initialized");
    
    // Initialize memory management first
    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    serial_println!("STEP 2: Physical memory offset: {:?}", phys_mem_offset);
    
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };
    serial_println!("STEP 3: Memory management initialized");
    
    // Disable interrupts during heap initialization
    x86_64::instructions::interrupts::disable();
    serial_println!("STEP 4: Interrupts disabled");
    
    // Initialize the heap
    match allocator::init_heap(&mut mapper, &mut frame_allocator) {
        Ok(_) => serial_println!("STEP 5: Heap initialization successful"),
        Err(e) => {
            serial_println!("ERROR: Heap initialization failed: {:?}", e);
            panic!("Failed to initialize heap");
        }
    }
    
    // Initialize basic components
    unia_os_bootable::gdt::init();
    serial_println!("STEP 6: GDT initialized");
    
    unia_os_bootable::interrupts::init_idt();
    serial_println!("STEP 7: IDT initialized");
    
    unsafe { unia_os_bootable::interrupts::PICS.lock().initialize() };
    serial_println!("STEP 8: PIC initialized");
    
    // Re-enable interrupts
    x86_64::instructions::interrupts::enable();
    serial_println!("STEP 9: Interrupts enabled");
    
    // Visual progress indicator
    println!("UNIA OS Kernel Starting...");
    serial_println!("STEP 10: VGA output initialized");
    
    // Test allocations to verify heap is working
    serial_println!("STEP 11: Testing allocations...");
    test_allocations();
    serial_println!("STEP 12: Allocation tests passed");
    
    // Try to initialize a minimal UI
    serial_println!("STEP 13: Attempting to initialize minimal UI...");
    println!("UNIA OS is ready!");
    println!("System is alive and running...");
    
    // Simple animation to show the system is alive
    serial_println!("STEP 14: Starting animation loop");
    let mut counter = 0;
    loop {
        // Update screen every 10 million cycles
        counter += 1;
        if counter % 10_000_000 == 0 {
            let progress = (counter / 10_000_000) % 10;
            let equals = "=".repeat(progress as usize);
            let spaces = " ".repeat(10 - progress as usize);
            let progress_str = equals + &spaces;
            
            println!("System alive: [{}>{}] {}", progress_str, " ".repeat(9 - progress as usize), counter / 10_000_000);
            serial_println!("HEARTBEAT: {}", counter / 10_000_000);
            
            // Update VGA buffer directly as a fallback
            unsafe {
                let vga_buffer = 0xb8000 as *mut u8;
                let message = b"ALIVE: ";
                
                for (i, &byte) in message.iter().enumerate() {
                    *vga_buffer.add(i * 2) = byte;
                    *vga_buffer.add(i * 2 + 1) = 0x0A; // Green on black
                }
                
                // Show counter as individual digits
                let count_val = counter / 10_000_000;
                let digit = (count_val % 10) as u8 + b'0';
                *vga_buffer.add(7 * 2) = digit;
                *vga_buffer.add(7 * 2 + 1) = 0x0A; // Green on black
            }
        }
        
        // Yield to CPU
        if counter % 1_000_000 == 0 {
            x86_64::instructions::hlt();
        }
    }
}

// Test different allocation sizes
fn test_allocations() {
    use alloc::vec::Vec;
    use alloc::string::String;
    
    // Test 1: Small allocation (16 bytes)
    serial_println!("Test 1: 16-byte allocation");
    let test1 = Vec::<u8>::with_capacity(16);
    serial_println!("16-byte allocation successful: {:p}", test1.as_ptr());
    
    // Test 2: Medium allocation (64 bytes) - this is the problematic one
    serial_println!("Test 2: 64-byte allocation");
    let test2 = Vec::<u8>::with_capacity(64);
    serial_println!("64-byte allocation successful: {:p}", test2.as_ptr());
    
    // Test 3: Large allocation (1024 bytes)
    serial_println!("Test 3: 1024-byte allocation");
    let test3 = Vec::<u8>::with_capacity(1024);
    serial_println!("1024-byte allocation successful: {:p}", test3.as_ptr());
    
    // Test 4: String allocation
    serial_println!("Test 4: String allocation");
    let test4 = String::from("UNIA OS String Test");
    serial_println!("String allocation successful: {:p} - {}", test4.as_ptr(), test4);
    
    // Prevent deallocation to avoid issues
    core::mem::forget(test1);
    core::mem::forget(test2);
    core::mem::forget(test3);
    core::mem::forget(test4);
}

/// This function is called on panic.
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("\n\nKERNEL PANIC: {}", info);
    
    // Print register state for debugging
    print_register_state();
    
    // Print additional debug information
    serial_println!("System halted.");
    
    hlt_loop();
}

// Print register state for debugging
fn print_register_state() {
    serial_println!("=== REGISTER STATE ===");
    
    let rax: u64;
    let rbx: u64;
    let rcx: u64;
    let rdx: u64;
    let rsp: u64;
    let rbp: u64;
    
    unsafe {
        core::arch::asm!("mov {}, rax", out(reg) rax);
        core::arch::asm!("mov {}, rbx", out(reg) rbx);
        core::arch::asm!("mov {}, rcx", out(reg) rcx);
        core::arch::asm!("mov {}, rdx", out(reg) rdx);
        core::arch::asm!("mov {}, rsp", out(reg) rsp);
        core::arch::asm!("mov {}, rbp", out(reg) rbp);
    }
    
    serial_println!("RAX: 0x{:016x}  RBX: 0x{:016x}", rax, rbx);
    serial_println!("RCX: 0x{:016x}  RDX: 0x{:016x}", rcx, rdx);
    serial_println!("RSP: 0x{:016x}  RBP: 0x{:016x}", rsp, rbp);
}
