#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Initialize VGA buffer directly without complex memory management
    init_vga_buffer();
    
    // Display welcome message
    print_string("UNIA OS v0.1.0 - Minimal Boot", 0, 0x0F);
    print_string("System Status: RUNNING", 1, 0x0A);
    print_string("Memory: Direct VGA access", 2, 0x0B);
    print_string("Display: 80x25 text mode", 3, 0x0C);
    print_string("", 4, 0x0F);
    print_string("This is a minimal OS without heap allocation", 5, 0x0E);
    print_string("No page table mapping required", 6, 0x0E);
    print_string("", 7, 0x0F);
    print_string("System initialized successfully!", 8, 0x0A);
    
    // Simple event loop
    loop {
        // Halt CPU to save power
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

fn init_vga_buffer() {
    // Clear screen
    let vga_buffer = 0xb8000 as *mut u8;
    for i in 0..(80 * 25 * 2) {
        unsafe {
            *vga_buffer.add(i) = 0;
        }
    }
}

fn print_string(s: &str, row: usize, color: u8) {
    let vga_buffer = 0xb8000 as *mut u8;
    let bytes = s.as_bytes();
    
    for (i, &byte) in bytes.iter().enumerate() {
        if i >= 80 { break; } // Don't exceed line width
        
        let offset = (row * 80 + i) * 2;
        unsafe {
            *vga_buffer.add(offset) = byte;
            *vga_buffer.add(offset + 1) = color;
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Display panic message directly to VGA buffer
    print_string("KERNEL PANIC!", 10, 0x4F); // White on red
    
    if let Some(location) = info.location() {
        print_string("Location:", 11, 0x0F);
        print_string(location.file(), 12, 0x0C);
    }
    
    if let Some(message) = info.message() {
        print_string("Message:", 13, 0x0F);
        // Note: Can't easily format the message without heap allocation
        print_string("Check serial output for details", 14, 0x0C);
    }
    
    // Halt forever
    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}
