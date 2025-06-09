#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{
    allocator, debug, hlt_loop, memory, println, serial_println
};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize the OS components first - minimal initialization
    unia_os_bootable::init();
    
    // Print debug information
    serial_println!("=== UNIA OS MINIMAL KERNEL ===");
    serial_println!("Starting with minimal initialization");
    
    // Initialize memory management immediately
    let phys_mem_offset = boot_info.physical_memory_offset;
    serial_println!("Physical memory offset: 0x{:x}", phys_mem_offset);
    
    let mut mapper = unsafe { memory::init(phys_mem_offset) };
    let mut frame_allocator = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_map)
    };
    
    // Now it's safe to print to VGA
    println!("UNIA OS Kernel Starting...");
    println!("Memory management initialized");
    
    // Initialize the heap
    println!("Initializing heap...");
    match allocator::init_heap() {
        Ok(_) => println!("Heap initialization successful"),
        Err(e) => {
            println!("Heap initialization failed: {}", e);
            panic!("Failed to initialize heap");
        }
    }
    
    // Test allocations
    println!("Testing allocations...");
    test_allocations();
    
    // If we get here, the allocations worked!
    println!("All allocations successful!");
    println!("UNIA OS is ready!");
    
    // Just halt - we're just testing allocations
    hlt_loop();
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
    println!("\n\nKERNEL PANIC: {}", info);
    serial_println!("\n\nKERNEL PANIC: {}", info);
    
    // Print register state for debugging
    debug::print_register_state();
    
    // Print additional debug information
    println!("System halted.");
    
    hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    unia_os_bootable::test_panic_handler(info)
}
