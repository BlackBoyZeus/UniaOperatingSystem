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
use pc_keyboard::{DecodedKey, KeyCode};

entry_point!(kernel_main);

// Menu options and screens
enum MenuScreen {
    Main,
    Dashboard,
    Terminal,
    Settings,
    Help,
}

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
    
    // Initialize keyboard
    let mut keyboard = unia_os_bootable::task::keyboard::init_keyboard();
    serial_println!("STEP 10: Keyboard initialized");
    
    // Initialize UI state
    let mut current_screen = MenuScreen::Main;
    let mut selected_option = 0;
    let mut counter = 0;
    
    // Draw the main screen
    draw_main_screen();
    serial_println!("STEP 11: Main screen drawn");
    
    // Enter the main event loop
    serial_println!("STEP 12: Entering main event loop");
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
        
        // Check for keyboard input
        if let Some(key) = keyboard.process_next_scancode() {
            serial_println!("Key pressed: {:?}", key);
            
            match key {
                DecodedKey::Unicode('1') => {
                    selected_option = 0;
                    highlight_menu_option(selected_option);
                    update_status_bar("Selected: System Dashboard");
                },
                DecodedKey::Unicode('2') => {
                    selected_option = 1;
                    highlight_menu_option(selected_option);
                    update_status_bar("Selected: Terminal");
                },
                DecodedKey::Unicode('3') => {
                    selected_option = 2;
                    highlight_menu_option(selected_option);
                    update_status_bar("Selected: Settings");
                },
                DecodedKey::Unicode('4') => {
                    selected_option = 3;
                    highlight_menu_option(selected_option);
                    update_status_bar("Selected: Help");
                },
                DecodedKey::Unicode('\n') | DecodedKey::Unicode('\r') => {
                    // Enter key pressed - navigate to selected screen
                    match selected_option {
                        0 => {
                            current_screen = MenuScreen::Dashboard;
                            draw_dashboard_screen();
                        },
                        1 => {
                            current_screen = MenuScreen::Terminal;
                            draw_terminal_screen();
                        },
                        2 => {
                            current_screen = MenuScreen::Settings;
                            draw_settings_screen();
                        },
                        3 => {
                            current_screen = MenuScreen::Help;
                            draw_help_screen();
                        },
                        _ => {}
                    }
                },
                DecodedKey::Unicode('\u{1b}') => { // Escape key
                    // Return to main menu
                    current_screen = MenuScreen::Main;
                    draw_main_screen();
                    selected_option = 0;
                    highlight_menu_option(selected_option);
                },
                _ => {}
            }
        }
        
        // Yield to CPU
        if counter % 100_000 == 0 {
            x86_64::instructions::hlt();
        }
    }
}

// Draw the main screen with menu options
fn draw_main_screen() {
    // Clear the screen completely
    clear_screen();
    
    // Draw a persistent UI frame
    draw_ui_frame();
    
    // Display welcome message
    display_centered_text("Welcome to UNIA OS", 5);
    display_centered_text("Unified Neural Interface Architecture", 6);
    display_centered_text("Version 0.1.0", 7);
    
    // Display system info
    display_text("System Information:", 10, 5);
    display_text("- CPU: Virtual x86_64", 11, 5);
    display_text("- Memory: 256 MB", 12, 5);
    display_text("- Storage: Virtual Disk", 13, 5);
    
    // Display menu options
    display_text("Menu Options:", 16, 5);
    display_text("1. System Dashboard", 17, 5);
    display_text("2. Terminal", 18, 5);
    display_text("3. Settings", 19, 5);
    display_text("4. Help", 20, 5);
    
    // Display status bar
    update_status_bar("System Ready | Press a key (1-4) to navigate");
    
    // Highlight the first option by default
    highlight_menu_option(0);
}

// Draw the dashboard screen
fn draw_dashboard_screen() {
    // Clear the screen completely
    clear_screen();
    
    // Draw a persistent UI frame
    draw_ui_frame();
    
    // Display dashboard title
    display_centered_text("UNIA OS Dashboard", 3);
    
    // Display system metrics
    display_text("System Metrics:", 5, 5);
    display_text("CPU Usage: 2%", 6, 5);
    display_text("Memory Usage: 24 MB / 256 MB", 7, 5);
    display_text("Disk Usage: 12 MB / 100 MB", 8, 5);
    
    // Display network status
    display_text("Network Status:", 10, 5);
    display_text("Status: Connected", 11, 5);
    display_text("Interface: Virtual NIC", 12, 5);
    display_text("IP Address: 192.168.1.100", 13, 5);
    
    // Display active processes
    display_text("Active Processes:", 15, 5);
    display_text("1. System Kernel", 16, 5);
    display_text("2. Terminal", 17, 5);
    display_text("3. Dashboard", 18, 5);
    
    // Display status bar
    update_status_bar("Dashboard | Press ESC to return to main menu");
}

// Draw the terminal screen
fn draw_terminal_screen() {
    // Clear the screen completely
    clear_screen();
    
    // Draw a persistent UI frame
    draw_ui_frame();
    
    // Display terminal title
    display_centered_text("UNIA OS Terminal", 3);
    
    // Display terminal prompt
    display_text("unia@os:~$ _", 5, 2);
    
    // Display some example output
    display_text("Welcome to the UNIA OS Terminal", 7, 2);
    display_text("Type 'help' for a list of commands", 8, 2);
    
    // Display status bar
    update_status_bar("Terminal | Press ESC to return to main menu");
}

// Draw the settings screen
fn draw_settings_screen() {
    // Clear the screen completely
    clear_screen();
    
    // Draw a persistent UI frame
    draw_ui_frame();
    
    // Display settings title
    display_centered_text("UNIA OS Settings", 3);
    
    // Display settings categories
    display_text("System Settings:", 5, 5);
    display_text("1. Display", 6, 5);
    display_text("   Resolution: 80x25", 7, 8);
    display_text("   Color Scheme: Blue", 8, 8);
    
    display_text("2. Network", 10, 5);
    display_text("   Auto-connect: Enabled", 11, 8);
    display_text("   DHCP: Enabled", 12, 8);
    
    display_text("3. Power", 14, 5);
    display_text("   Sleep after: 30 minutes", 15, 8);
    display_text("   Performance mode: Balanced", 16, 8);
    
    // Display status bar
    update_status_bar("Settings | Press ESC to return to main menu");
}

// Draw the help screen
fn draw_help_screen() {
    // Clear the screen completely
    clear_screen();
    
    // Draw a persistent UI frame
    draw_ui_frame();
    
    // Display help title
    display_centered_text("UNIA OS Help", 3);
    
    // Display help content
    display_text("Navigation:", 5, 5);
    display_text("- Use number keys (1-4) to select menu options", 6, 5);
    display_text("- Press Enter to navigate to selected option", 7, 5);
    display_text("- Press ESC to return to the main menu", 8, 5);
    
    display_text("About UNIA OS:", 10, 5);
    display_text("UNIA OS (Unified Neural Interface Architecture) is a", 11, 5);
    display_text("next-generation operating system designed for advanced", 12, 5);
    display_text("AI integration and neural interface capabilities.", 13, 5);
    
    display_text("For more information, visit:", 15, 5);
    display_text("https://github.com/BlackBoyZeus/UniaOperatingSystem", 16, 5);
    
    // Display status bar
    update_status_bar("Help | Press ESC to return to main menu");
}

// Highlight the selected menu option
fn highlight_menu_option(option: usize) {
    // Reset all options to normal
    for i in 0..4 {
        let row = 17 + i;
        let color = 0x07; // White on black
        
        unsafe {
            let vga_buffer = 0xb8000 as *mut u8;
            for col in 0..20 {
                *vga_buffer.add((row * 80 + col + 5) * 2 + 1) = color;
            }
        }
    }
    
    // Highlight the selected option
    let row = 17 + option;
    let color = 0x70; // Black on white
    
    unsafe {
        let vga_buffer = 0xb8000 as *mut u8;
        for col in 0..20 {
            *vga_buffer.add((row * 80 + col + 5) * 2 + 1) = color;
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
