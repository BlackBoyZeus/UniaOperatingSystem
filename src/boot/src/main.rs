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
    
    // Write directly to serial port for debugging
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
    
    // Pre-allocate some memory for critical components
    serial_println!("Pre-allocating critical memory...");
    let layouts = [
        Layout::from_size_align(64, 8).unwrap(),
        Layout::from_size_align(128, 8).unwrap(),
        Layout::from_size_align(256, 8).unwrap(),
    ];
    
    let mut critical_ptrs = [core::ptr::null_mut(); 3];
    for (i, layout) in layouts.iter().enumerate() {
        unsafe {
            critical_ptrs[i] = alloc::alloc::alloc(layout.clone());
            if critical_ptrs[i].is_null() {
                serial_println!("Critical pre-allocation failed for size {}, align {}", 
                               layout.size(), layout.align());
            } else {
                serial_println!("Critical pre-allocation successful: {:p}", critical_ptrs[i]);
            }
        }
    }
    
    // Now initialize basic OS components
    serial_println!("Initializing basic OS components...");
    unia_os_bootable::gdt::init();
    unia_os_bootable::interrupts::init_idt();
    
    // Initialize PIC and enable interrupts only after heap is ready
    serial_println!("Initializing PIC...");
    unsafe { unia_os_bootable::interrupts::PICS.lock().initialize() };
    
    serial_println!("Re-enabling interrupts");
    x86_64::instructions::interrupts::enable();
    
    // Test manual allocation to verify allocator
    serial_println!("Testing manual allocation...");
    let layout = Layout::from_size_align(64, 8).unwrap();
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        serial_println!("Manual allocation failed for size 64, align 8");
    } else {
        serial_println!("Manual allocation successful: {:p}", ptr);
        unsafe { alloc::alloc::dealloc(ptr, layout) };
    }
    
    // Now it's safe to use println
    serial_println!("Before println! call");
    println!("UNIA OS Kernel Starting...");
    serial_println!("After first println! call");
    println!("Memory management initialized");
    serial_println!("After second println! call");
    
    // Test allocations
    serial_println!("Starting test allocations");
    test_allocations();
    
    // If we get here, the allocations worked!
    serial_println!("All allocations successful!");
    println!("All allocations successful!");
    println!("UNIA OS is ready!");
    
    // Free pre-allocated memory
    for (i, layout) in layouts.iter().enumerate() {
        if !critical_ptrs[i].is_null() {
            unsafe { alloc::alloc::dealloc(critical_ptrs[i], layout.clone()); }
        }
    }
    
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
