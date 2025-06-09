#![no_std]
#![no_main]

use core::panic::PanicInfo;
use bootloader::{entry_point, BootInfo};

entry_point!(kernel_main);

const VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;
const SCREEN_WIDTH: usize = 80;
const SCREEN_HEIGHT: usize = 25;

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Clear screen
    for i in 0..(SCREEN_WIDTH * SCREEN_HEIGHT * 2) {
        unsafe {
            *VGA_BUFFER.add(i) = 0;
        }
    }

    // Print welcome message
    let msg = "UNIA OS v0.1.0 - Standalone Mode";
    for (i, &byte) in msg.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add(i * 2) = byte;
            *VGA_BUFFER.add(i * 2 + 1) = 0x0F; // White on black
        }
    }

    // Print status message
    let status = "System Status: OK";
    for (i, &byte) in status.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add((SCREEN_WIDTH * 2) + i * 2) = byte;
            *VGA_BUFFER.add((SCREEN_WIDTH * 2) + i * 2 + 1) = 0x0A; // Green on black
        }
    }

    // Print memory message
    let mem = "Memory: Direct VGA Mode";
    for (i, &byte) in mem.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add((SCREEN_WIDTH * 4) + i * 2) = byte;
            *VGA_BUFFER.add((SCREEN_WIDTH * 4) + i * 2 + 1) = 0x0B; // Light cyan on black
        }
    }

    // Print info message
    let info = "No page tables or complex memory management required";
    for (i, &byte) in info.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add((SCREEN_WIDTH * 6) + i * 2) = byte;
            *VGA_BUFFER.add((SCREEN_WIDTH * 6) + i * 2 + 1) = 0x0E; // Yellow on black
        }
    }

    // Print ready message
    let ready = "System ready - Press any key to continue";
    for (i, &byte) in ready.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add((SCREEN_WIDTH * 8) + i * 2) = byte;
            *VGA_BUFFER.add((SCREEN_WIDTH * 8) + i * 2 + 1) = 0x0F; // White on black
        }
    }

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // Print panic message in red
    let msg = "KERNEL PANIC!";
    for (i, &byte) in msg.as_bytes().iter().enumerate() {
        unsafe {
            *VGA_BUFFER.add((SCREEN_WIDTH * 12) + i * 2) = byte;
            *VGA_BUFFER.add((SCREEN_WIDTH * 12) + i * 2 + 1) = 0x4F; // White on red
        }
    }

    loop {
        unsafe {
            core::arch::asm!("hlt");
        }
    }
}
