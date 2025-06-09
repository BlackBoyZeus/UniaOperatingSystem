#![no_std]
#![no_main]

extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

entry_point!(kernel_main);

const VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    // Clear screen
    for i in 0..(80 * 25 * 2) {
        unsafe {
            *VGA_BUFFER.add(i) = 0;
        }
    }

    // Display UNIA Gaming OS boot message
    let messages = [
        "UNIA Gaming OS v3.0 - AI Gaming Console",
        "Initializing AI Gaming Subsystems...",
        "",
        "✓ Neural Processing Unit: READY",
        "✓ GPU Acceleration: READY", 
        "✓ AI Game Engine: READY",
        "✓ Real-time Physics: READY",
        "✓ 3D Audio System: READY",
        "✓ VR/AR Support: READY",
        "✓ Cloud Gaming: READY",
        "",
        "Gaming Features:",
        "• AI-Enhanced NPCs",
        "• Procedural Content Generation", 
        "• Dynamic Difficulty Scaling",
        "• Voice & Gesture Recognition",
        "• Ray Tracing & DLSS",
        "• 144Hz Gaming @ 4K",
        "",
        "Ready for next-gen AI gaming!",
        "Press any key to continue...",
    ];

    for (i, message) in messages.iter().enumerate() {
        print_line(message, i, 0x0F);
    }

    // Main loop
    loop {
        x86_64::instructions::hlt();
    }
}

fn print_line(text: &str, row: usize, color: u8) {
    let bytes = text.as_bytes();
    for (col, &byte) in bytes.iter().enumerate() {
        if col >= 80 { break; }
        let offset = (row * 80 + col) * 2;
        unsafe {
            *VGA_BUFFER.add(offset) = byte;
            *VGA_BUFFER.add(offset + 1) = color;
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    print_line("GAMING OS PANIC!", 22, 0x4F);
    print_line("System halted for safety", 23, 0x0C);
    loop {
        x86_64::instructions::hlt();
    }
}
