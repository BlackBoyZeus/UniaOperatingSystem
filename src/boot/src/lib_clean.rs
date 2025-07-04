#![no_std]
#![cfg_attr(test, no_main)]
#![feature(custom_test_frameworks)]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]

extern crate alloc;

pub mod allocator;
pub mod console;
pub mod gdt;
pub mod interrupts;
pub mod memory;
pub mod serial;
pub mod task;
pub mod ui;
pub mod vga_buffer;
pub mod ai;
pub mod network;
pub mod game;

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, Ordering};

// Flag to track if heap is initialized
static HEAP_INITIALIZED: AtomicBool = AtomicBool::new(false);

// Function to mark heap as initialized
pub fn mark_heap_initialized() {
    HEAP_INITIALIZED.store(true, Ordering::SeqCst);
}

// Function to check if heap is initialized
pub fn is_heap_initialized() -> bool {
    HEAP_INITIALIZED.load(Ordering::SeqCst)
}

#[cfg(test)]
use bootloader::{entry_point, BootInfo};

#[cfg(test)]
entry_point!(test_kernel_main);

/// Entry point for `cargo test`
#[cfg(test)]
fn test_kernel_main(_boot_info: &'static BootInfo) -> ! {
    init();
    test_main();
    hlt_loop();
}

pub fn init() {
    serial_println!("Initializing GDT...");
    gdt::init();
    
    serial_println!("Initializing IDT...");
    interrupts::init_idt();
    
    serial_println!("Initializing PIC...");
    unsafe { interrupts::PICS.lock().initialize() };
    
    serial_println!("Enabling interrupts...");
    x86_64::instructions::interrupts::enable();
    
    serial_println!("Basic initialization complete");
}

pub fn hlt_loop() -> ! {
    loop {
        x86_64::instructions::hlt();
    }
}

pub trait Testable {
    fn run(&self) -> ();
}

impl<T> Testable for T
where
    T: Fn(),
{
    fn run(&self) {
        serial_print!("{}...\t", core::any::type_name::<T>());
        self();
        serial_println!("[ok]");
    }
}

pub fn test_runner(tests: &[&dyn Testable]) {
    serial_println!("Running {} tests", tests.len());
    for test in tests {
        test.run();
    }
    exit_qemu(QemuExitCode::Success);
}

pub fn test_panic_handler(info: &PanicInfo) -> ! {
    serial_println!("[failed]\n");
    serial_println!("Error: {}\n", info);
    exit_qemu(QemuExitCode::Failed);
    hlt_loop();
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    test_panic_handler(info)
}

#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    serial_println!("=== ALLOCATION ERROR DETECTED ===");
    serial_println!("Layout: size={}, align={}", layout.size(), layout.align());
    
    if !is_heap_initialized() {
        serial_println!("Heap not initialized yet!");
    }
    
    serial_println!("=== SYSTEM HALTED ===");
    
    panic!(
        "Allocation error: {:?} - size: {}, align: {}",
        layout, layout.size(), layout.align()
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

pub fn exit_qemu(exit_code: QemuExitCode) {
    use x86_64::instructions::port::Port;

    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
}
