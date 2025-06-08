use crate::{println, vga_buffer};
use core::fmt::Write;
use lazy_static::lazy_static;
use spin::Mutex;
use vga::colors::{Color16, TextModeColor};

// UNIA OS Boot Animation frames
const BOOT_FRAMES: [&str; 5] = [
    r#"
    █    █ █▄ █ █ ▄▀█    █▀█ █▀
    █    █ █ ▀█ █ █▀█    █▄█ ▄█
    "#,
    r#"
    █    █ █▄ █ █ ▄▀█    █▀█ █▀
    █    █ █ ▀█ █ █▀█ ▄  █▄█ ▄█
    "#,
    r#"
    █    █ █▄ █ █ ▄▀█    █▀█ █▀
    █ ▄  █ █ ▀█ █ █▀█ ▄  █▄█ ▄█
    "#,
    r#"
    █    █ █▄ █ █ ▄▀█    █▀█ █▀
    █ ▄▄ █ █ ▀█ █ █▀█ ▄  █▄█ ▄█
    "#,
    r#"
    █    █ █▄ █ █ ▄▀█    █▀█ █▀
    █ ▄▄▄█ █ ▀█ █ █▀█ ▄  █▄█ ▄█
    "#,
];

const UNIA_LOGO: &str = r#"
██╗   ██╗███╗   ██╗██╗ █████╗      ██████╗ ███████╗
██║   ██║████╗  ██║██║██╔══██╗    ██╔═══██╗██╔════╝
██║   ██║██╔██╗ ██║██║███████║    ██║   ██║███████╗
██║   ██║██║╚██╗██║██║██╔══██║    ██║   ██║╚════██║
╚██████╔╝██║ ╚████║██║██║  ██║    ╚██████╔╝███████║
 ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝     ╚═════╝ ╚══════╝
"#;

const POWERED_BY_TEXT: &str = "Powered by Unified Neural Interface Architecture";
const COPYRIGHT_TEXT: &str = "© 2025 UNIA OS Team. All rights reserved.";
const VERSION_TEXT: &str = "UNIA OS v1.0.0";

pub fn run_boot_sequence() {
    // Clear screen
    clear_screen();
    
    // Display boot animation
    for frame in BOOT_FRAMES.iter() {
        clear_screen();
        center_text(frame, 10);
        delay(500_000);
    }
    
    // Display UNIA logo
    clear_screen();
    set_color(vga_buffer::Color::LightGreen, vga_buffer::Color::Black);
    center_text(UNIA_LOGO, 5);
    
    // Display powered by text
    set_color(vga_buffer::Color::White, vga_buffer::Color::Black);
    center_text(POWERED_BY_TEXT, 15);
    
    // Display version and copyright
    set_color(vga_buffer::Color::LightGray, vga_buffer::Color::Black);
    center_text(VERSION_TEXT, 20);
    center_text(COPYRIGHT_TEXT, 22);
    
    // Loading bar
    display_loading_bar(24);
    
    // Reset color
    set_color(vga_buffer::Color::White, vga_buffer::Color::Black);
    
    // Delay before continuing to main OS
    delay(1_000_000);
    
    // Clear screen before continuing
    clear_screen();
}

fn clear_screen() {
    let mut writer = vga_buffer::WRITER.lock();
    writer.clear_screen();
}

fn center_text(text: &str, row: usize) {
    let lines: Vec<&str> = text.lines().collect();
    let mut current_row = row;
    
    for line in lines {
        if !line.is_empty() {
            let padding = (80 - line.len()) / 2;
            let mut writer = vga_buffer::WRITER.lock();
            
            // Position cursor
            writer.set_position(current_row, 0);
            
            // Write padding
            for _ in 0..padding {
                write!(writer, " ").unwrap();
            }
            
            // Write text
            write!(writer, "{}", line).unwrap();
        }
        current_row += 1;
    }
}

fn display_loading_bar(row: usize) {
    let bar_width = 50;
    let mut writer = vga_buffer::WRITER.lock();
    
    // Position cursor
    writer.set_position(row, 0);
    
    // Write padding
    let padding = (80 - bar_width - 2) / 2;
    for _ in 0..padding {
        write!(writer, " ").unwrap();
    }
    
    // Draw empty bar
    write!(writer, "[").unwrap();
    for _ in 0..bar_width {
        write!(writer, " ").unwrap();
    }
    write!(writer, "]").unwrap();
    
    // Fill the bar
    for i in 0..=bar_width {
        // Reposition cursor
        writer.set_position(row, padding + 1);
        
        // Draw filled portion
        set_color(vga_buffer::Color::LightBlue, vga_buffer::Color::Black);
        for _ in 0..i {
            write!(writer, "█").unwrap();
        }
        
        delay(100_000);
    }
}

fn set_color(foreground: vga_buffer::Color, background: vga_buffer::Color) {
    let mut writer = vga_buffer::WRITER.lock();
    writer.color_code = vga_buffer::ColorCode::new(foreground, background);
}

fn delay(count: u64) {
    for _ in 0..count {
        // Simple delay loop
        core::hint::spin_loop();
    }
}
