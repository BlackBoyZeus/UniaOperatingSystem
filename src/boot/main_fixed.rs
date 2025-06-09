#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(unia_os_bootable::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use unia_os_bootable::{hlt_loop, serial_println};
use x86_64::VirtAddr;

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize serial output first
    unia_os_bootable::serial::init();
    serial_println!("UNIA OS Starting...");
    
    // Initialize VGA buffer for display
    unia_os_bootable::vga_buffer::init();
    
    // Print welcome message
    println!("Welcome to UNIA Operating System!");
    println!("Initializing system components...");
    
    // Initialize GDT
    unia_os_bootable::gdt::init();
    serial_println!("GDT initialized");
    
    // Initialize IDT
    unia_os_bootable::interrupts::init_idt();
    serial_println!("IDT initialized");
    
    // Initialize PIC
    unsafe { unia_os_bootable::interrupts::PICS.lock().initialize() };
    serial_println!("PIC initialized");
    
    // Enable interrupts
    x86_64::instructions::interrupts::enable();
    serial_println!("Interrupts enabled");
    
    // Display main menu
    display_main_menu();
    
    // Main event loop
    loop {
        x86_64::instructions::hlt();
    }
}

fn display_main_menu() {
    println!("");
    println!("╔══════════════════════════════════════╗");
    println!("║           UNIA OS v0.1.0             ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  1. System Information               ║");
    println!("║  2. Terminal                         ║");
    println!("║  3. Settings                         ║");
    println!("║  4. Help                             ║");
    println!("║  5. Shutdown                         ║");
    println!("╚══════════════════════════════════════╝");
    println!("");
    println!("Use arrow keys to navigate, Enter to select");
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("PANIC: {}", info);
    println!("PANIC: {}", info);
    hlt_loop();
}
