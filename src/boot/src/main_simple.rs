#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{hlt_loop, serial_println, println};

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize serial output first
    unia_os_bootable::serial::init();
    serial_println!("UNIA OS Starting...");
    
    // Initialize VGA buffer for display
    unia_os_bootable::vga_buffer::init();
    
    // Clear screen and display welcome message
    clear_screen();
    println!("╔══════════════════════════════════════╗");
    println!("║           UNIA OS v0.1.0             ║");
    println!("║        Minimal Boot Version          ║");
    println!("╚══════════════════════════════════════╝");
    println!("");
    println!("System Status: RUNNING");
    println!("Memory: Basic initialization complete");
    println!("Display: VGA text mode active");
    println!("");
    
    // Initialize basic system components
    unia_os_bootable::gdt::init();
    println!("✓ GDT initialized");
    
    unia_os_bootable::interrupts::init_idt();
    println!("✓ IDT initialized");
    
    unsafe { unia_os_bootable::interrupts::PICS.lock().initialize() };
    println!("✓ PIC initialized");
    
    x86_64::instructions::interrupts::enable();
    println!("✓ Interrupts enabled");
    
    println!("");
    println!("UNIA OS is now running!");
    println!("This is a minimal version without heap allocation.");
    println!("Press any key to see keyboard input...");
    
    // Main event loop
    loop {
        x86_64::instructions::hlt();
    }
}

fn clear_screen() {
    for _ in 0..25 {
        println!("");
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("PANIC: {}", info);
    println!("PANIC: {}", info);
    hlt_loop();
}
