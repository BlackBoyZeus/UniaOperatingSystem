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
use core::alloc::Layout;

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Direct VGA buffer manipulation (no allocations)
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        let message = b"UNIA OS Starting...";
        
        for (i, &byte) in message.iter().enumerate() {
            *vga_buffer.add(i * 2) = byte;
            *vga_buffer.add(i * 2 + 1) = 0x0F; // White on black
        }
    }
    
    serial_println!("UNIA OS Serial Initialized - Direct Write");
    
    // Initialize memory management first
    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    serial_println!("Physical memory offset: {:?}", phys_mem_offset);
    
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };
    serial_println!("Memory management initialized");
    
    // Disable interrupts during heap initialization to prevent interference
    serial_println!("Disabling interrupts for heap initialization");
    x86_64::instructions::interrupts::disable();
    
    // Initialize the heap - this will set up the main allocator
    serial_println!("Initializing heap...");
    match allocator::init_heap(&mut mapper, &mut frame_allocator) {
        Ok(_) => serial_println!("Heap initialization successful"),
        Err(e) => {
            serial_println!("Heap initialization failed: {:?}", e);
            panic!("Failed to initialize heap");
        }
    }
    
    // Initialize basic components
    serial_println!("Initializing GDT...");
    unia_os_bootable::gdt::init();
    serial_println!("GDT initialized");
    
    serial_println!("Initializing IDT...");
    unia_os_bootable::interrupts::init_idt();
    serial_println!("IDT initialized");
    
    serial_println!("Initializing PIC...");
    unsafe { unia_os_bootable::interrupts::PICS.lock().initialize() };
    serial_println!("PIC initialized");
    
    // Re-enable interrupts
    serial_println!("Re-enabling interrupts");
    x86_64::instructions::interrupts::enable();
    serial_println!("Interrupts enabled");
    
    // Now it's safe to use println
    println!("UNIA OS Kernel Starting...");
    println!("Memory management initialized");
    
    // Test allocations to verify heap is working
    serial_println!("Testing allocations...");
    test_allocations();
    serial_println!("Allocation tests passed");
    
    // Try to initialize a minimal UI
    serial_println!("Attempting to initialize minimal UI...");
    println!("UNIA OS is ready!");
    println!("Press any key to continue...");
    
    // Try to initialize keyboard handling
    serial_println!("Initializing keyboard handler...");
    let mut keyboard = unia_os_bootable::task::keyboard::init_keyboard();
    serial_println!("Keyboard handler initialized");
    
    // Try to initialize a simple task executor
    serial_println!("Initializing simple task executor...");
    
    // Enter a simple event loop that just responds to keypresses
    serial_println!("Entering simple event loop");
    loop {
        // Process any pending keyboard events
        if let Some(key) = keyboard.process_next_scancode() {
            serial_println!("Key pressed: {:?}", key);
            println!("Key pressed: {:?}", key);
        }
        
        // Yield to CPU
        x86_64::instructions::hlt();
    }
}

// Test different allocation sizes
fn test_allocations() {
    use alloc::vec::Vec;
    
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
    
    // Prevent deallocation to avoid issues
    core::mem::forget(test1);
    core::mem::forget(test2);
    core::mem::forget(test3);
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
