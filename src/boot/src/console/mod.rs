use crate::{println, vga_buffer, print};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use lazy_static::lazy_static;
use spin::Mutex;

// Console state
lazy_static! {
    static ref CONSOLE: Mutex<Console> = Mutex::new(Console::new());
}

pub struct Console {
    title: String,
    prompt: String,
    command_history: Vec<String>,
    current_command: String,
    cursor_position: usize,
}

impl Console {
    pub fn new() -> Self {
        Console {
            title: String::from("UNIA OS Console"),
            prompt: String::from("unia> "),
            command_history: Vec::new(),
            current_command: String::new(),
            cursor_position: 0,
        }
    }

    pub fn init(&mut self) {
        self.clear_screen();
        self.draw_header();
        self.draw_prompt();
    }

    pub fn clear_screen(&self) {
        let mut writer = vga_buffer::WRITER.lock();
        writer.clear_screen();
    }

    pub fn draw_header(&self) {
        let mut writer = vga_buffer::WRITER.lock();
        writer.set_position(0, 0);
        
        // Set color for header
        writer.color_code = vga_buffer::ColorCode::new(
            vga_buffer::Color::Black,
            vga_buffer::Color::LightGreen
        );
        
        // Draw header bar
        for _ in 0..80 {
            write!(writer, " ").unwrap();
        }
        
        // Position for title
        writer.set_position(0, (80 - self.title.len()) / 2);
        write!(writer, "{}", self.title).unwrap();
        
        // Reset color
        writer.color_code = vga_buffer::ColorCode::new(
            vga_buffer::Color::LightGreen,
            vga_buffer::Color::Black
        );
    }

    pub fn draw_prompt(&self) {
        println!("\n{}{}", self.prompt, self.current_command);
    }

    pub fn process_key(&mut self, key: char) {
        match key {
            '\n' => self.execute_command(),
            '\x08' => self.backspace(), // Backspace
            _ => {
                self.current_command.push(key);
                self.cursor_position += 1;
                print!("{}", key);
            }
        }
    }

    fn backspace(&mut self) {
        if self.cursor_position > 0 {
            self.current_command.pop();
            self.cursor_position -= 1;
            print!("\x08 \x08"); // Backspace, space, backspace
        }
    }

    fn execute_command(&mut self) {
        let command = self.current_command.clone();
        self.command_history.push(command.clone());
        
        println!();
        
        // Process command
        match command.as_str() {
            "help" => self.cmd_help(),
            "clear" => self.cmd_clear(),
            "version" => self.cmd_version(),
            "reboot" => self.cmd_reboot(),
            "shutdown" => self.cmd_shutdown(),
            "games" => self.cmd_games(),
            "network" => self.cmd_network(),
            "ai" => self.cmd_ai(),
            "" => {}, // Do nothing for empty command
            _ => println!("Unknown command: {}", command),
        }
        
        // Reset current command
        self.current_command.clear();
        self.cursor_position = 0;
        
        // Draw new prompt
        self.draw_prompt();
    }

    // Command implementations
    fn cmd_help(&self) {
        println!("Available commands:");
        println!("  help      - Show this help message");
        println!("  clear     - Clear the screen");
        println!("  version   - Show UNIA OS version");
        println!("  reboot    - Reboot the system");
        println!("  shutdown  - Shutdown the system");
        println!("  games     - List available games");
        println!("  network   - Show network status");
        println!("  ai        - Show AI subsystem status");
    }

    fn cmd_clear(&mut self) {
        self.clear_screen();
        self.draw_header();
    }

    fn cmd_version(&self) {
        println!("UNIA OS v1.0.0");
        println!("Unified Neural Interface Architecture");
        println!("© 2025 UNIA OS Team. All rights reserved.");
    }

    fn cmd_reboot(&self) {
        println!("Rebooting system...");
        // In a real implementation, this would trigger a reboot
    }

    fn cmd_shutdown(&self) {
        println!("Shutting down system...");
        // In a real implementation, this would trigger a shutdown
    }

    fn cmd_games(&self) {
        println!("Available games:");
        println!("  1. UNIA Demo Game");
        println!("  2. AI Sandbox");
        println!("  3. Mesh Network Test");
    }

    fn cmd_network(&self) {
        println!("Network status:");
        println!("  Status: Online");
        println!("  Mesh nodes: 3");
        println!("  Bandwidth: 5.4 Mbps");
        println!("  Latency: 12ms");
    }

    fn cmd_ai(&self) {
        println!("AI subsystem status:");
        println!("  Status: Active");
        println!("  Models loaded: 3");
        println!("  NPC behaviors: Active");
        println!("  Procedural generation: Ready");
    }
}

pub fn init_console() {
    let mut console = CONSOLE.lock();
    console.init();
}

pub fn process_key(key: char) {
    let mut console = CONSOLE.lock();
    console.process_key(key);
}
