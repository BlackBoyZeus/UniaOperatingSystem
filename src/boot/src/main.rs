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
use alloc::vec::Vec;

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
    
    // Clear the screen completely
    clear_screen();
    serial_println!("STEP 10: Screen cleared");
    
    // Draw a persistent UI frame
    draw_ui_frame();
    serial_println!("STEP 11: UI frame drawn");
    
    // Display welcome message
    display_centered_text("Welcome to UNIA OS", 5);
    display_centered_text("Universal Neural Intelligence Architecture", 6);
    display_centered_text("Version 0.1.0", 7);
    serial_println!("STEP 12: Welcome message displayed");
    
    // Display system info
    display_text("System Information:", 10, 5);
    display_text("- CPU: Virtual x86_64", 11, 5);
    display_text("- Memory: 256 MB", 12, 5);
    display_text("- Storage: Virtual Disk", 13, 5);
    serial_println!("STEP 13: System info displayed");
    
    // Display menu options
    display_text("Menu Options:", 16, 5);
    display_text("1. System Dashboard", 17, 5);
    display_text("2. Terminal", 18, 5);
    display_text("3. Settings", 19, 5);
    display_text("4. Help", 20, 5);
    serial_println!("STEP 14: Menu options displayed");
    
    // Display status bar
    update_status_bar("System Ready | Press a key (1-4) to navigate");
    serial_println!("STEP 15: Status bar displayed");
    
    // Enter the main event loop
    serial_println!("STEP 16: Entering main event loop");
    let mut counter = 0;
    let mut selected_option = 0;
    
    loop {
        counter += 1;
        
        // Update the clock every second (approximately)
        if counter % 10_000_000 == 0 {
            let time_value = counter / 10_000_000;
            let hours = (time_value / 3600) % 24;
            let minutes = (time_value / 60) % 60;
            let seconds = time_value % 60;
            
            // Format time as HH:MM:SS
            let time_str = format!("{:02}:{:02}:{:02}", hours, minutes, seconds);
            display_text(&time_str, 1, 70);
            
            serial_println!("Heartbeat: {}", time_value);
        }
        
        // Check for keyboard input (simulated for now)
        if counter % 50_000_000 == 0 {
            // Simulate menu selection
            selected_option = (selected_option + 1) % 4;
            
            // Highlight the selected option
            for i in 0..4 {
                let row = 17 + i;
                let highlight = if i == selected_option { 0x70 } else { 0x07 }; // White on black or black on white
                
                // Update the color attribute for the entire line
                unsafe {
                    let vga_buffer = 0xb8000 as *mut u8;
                    for col in 0..20 {
                        *vga_buffer.add((row * 80 + col + 5) * 2 + 1) = highlight;
                    }
                }
            }
            
            // Update status message
            let status_msg = format!("Selected option: {} | System uptime: {} seconds", 
                                    selected_option + 1, counter / 10_000_000);
            update_status_bar(&status_msg);
        }
        
        // Yield to CPU
        if counter % 100_000 == 0 {
            x86_64::instructions::hlt();
        }
    }
}

// Clear the entire screen
fn clear_screen() {
    let blank = 0x0720; // Space character with white on black attribute
    
    unsafe {
        let vga_buffer = 0xb8000 as *mut u16;
        for i in 0..(80 * 25) {
            *vga_buffer.add(i) = blank;
        }
    }
}

// Draw a UI frame around the screen
fn draw_ui_frame() {
    let width = 80;
    let height = 25;
    
    // Characters for the frame
    let top_left = 0xC9;     // ┌
    let top_right = 0xBB;    // ┐
    let bottom_left = 0xC8;  // └
    let bottom_right = 0xBC; // ┘
    let horizontal = 0xCD;   // ═
    let vertical = 0xBA;     // ║
    
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        
        // Draw top border
        *vga_buffer.add(0 * 2) = top_left as u8;
        *vga_buffer.add(0 * 2 + 1) = 0x1F; // Blue on white
        
        for i in 1..(width - 1) {
            *vga_buffer.add(i * 2) = horizontal as u8;
            *vga_buffer.add(i * 2 + 1) = 0x1F; // Blue on white
        }
        
        *vga_buffer.add((width - 1) * 2) = top_right as u8;
        *vga_buffer.add((width - 1) * 2 + 1) = 0x1F; // Blue on white
        
        // Draw bottom border
        *vga_buffer.add((height - 1) * width * 2) = bottom_left as u8;
        *vga_buffer.add((height - 1) * width * 2 + 1) = 0x1F; // Blue on white
        
        for i in 1..(width - 1) {
            *vga_buffer.add(((height - 1) * width + i) * 2) = horizontal as u8;
            *vga_buffer.add(((height - 1) * width + i) * 2 + 1) = 0x1F; // Blue on white
        }
        
        *vga_buffer.add(((height - 1) * width + width - 1) * 2) = bottom_right as u8;
        *vga_buffer.add(((height - 1) * width + width - 1) * 2 + 1) = 0x1F; // Blue on white
        
        // Draw left and right borders
        for i in 1..(height - 1) {
            *vga_buffer.add((i * width) * 2) = vertical as u8;
            *vga_buffer.add((i * width) * 2 + 1) = 0x1F; // Blue on white
            
            *vga_buffer.add((i * width + width - 1) * 2) = vertical as u8;
            *vga_buffer.add((i * width + width - 1) * 2 + 1) = 0x1F; // Blue on white
        }
        
        // Draw title
        let title = b"UNIA OS";
        let title_pos = (width - title.len()) / 2;
        
        for (i, &byte) in title.iter().enumerate() {
            *vga_buffer.add((title_pos + i) * 2) = byte;
            *vga_buffer.add((title_pos + i) * 2 + 1) = 0x1F; // Blue on white
        }
        
        // Draw status bar separator (second to last row)
        for i in 0..width {
            *vga_buffer.add(((height - 2) * width + i) * 2) = horizontal as u8;
            *vga_buffer.add(((height - 2) * width + i) * 2 + 1) = 0x1F; // Blue on white
        }
    }
}

// Display text at a specific position
fn display_text(text: &str, row: usize, col: usize) {
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        
        for (i, byte) in text.bytes().enumerate() {
            *vga_buffer.add(((row * 80) + col + i) * 2) = byte;
            *vga_buffer.add(((row * 80) + col + i) * 2 + 1) = 0x07; // White on black
        }
    }
}

// Display centered text
fn display_centered_text(text: &str, row: usize) {
    let col = (80 - text.len()) / 2;
    display_text(text, row, col);
}

// Update the status bar at the bottom of the screen
fn update_status_bar(text: &str) {
    let row = 24; // Last row
    let col = 2;  // Slight indent
    
    // Clear the status bar first
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        
        for i in 0..76 {
            *vga_buffer.add(((row * 80) + col + i) * 2) = b' ';
            *vga_buffer.add(((row * 80) + col + i) * 2 + 1) = 0x17; // White on blue
        }
        
        // Display the new status text
        for (i, byte) in text.bytes().enumerate() {
            if i >= 76 { break; } // Prevent overflow
            *vga_buffer.add(((row * 80) + col + i) * 2) = byte;
            *vga_buffer.add(((row * 80) + col + i) * 2 + 1) = 0x17; // White on blue
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
    
    // Display panic message on screen
    clear_screen();
    display_centered_text("KERNEL PANIC", 10);
    display_centered_text(&format!("{}", info), 12);
    display_centered_text("System Halted", 14);
    
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
