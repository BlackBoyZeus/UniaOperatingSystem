use core::fmt;
use lazy_static::lazy_static;
use spin::Mutex;
use volatile::Volatile;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    Blue = 1,
    Green = 2,
    Cyan = 3,
    Red = 4,
    Magenta = 5,
    Brown = 6,
    LightGray = 7,
    DarkGray = 8,
    LightBlue = 9,
    LightGreen = 10,
    LightCyan = 11,
    LightRed = 12,
    Pink = 13,
    Yellow = 14,
    White = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ColorCode(u8);

impl ColorCode {
    pub fn new(foreground: Color, background: Color) -> ColorCode {
        ColorCode((background as u8) << 4 | (foreground as u8))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct ScreenChar {
    ascii_character: u8,
    color_code: ColorCode,
}

const BUFFER_HEIGHT: usize = 25;
const BUFFER_WIDTH: usize = 80;

// Define a custom Volatile wrapper for ScreenChar
struct VolatileScreenChar {
    inner: ScreenChar,
}

impl VolatileScreenChar {
    fn new(ch: ScreenChar) -> Self {
        VolatileScreenChar { inner: ch }
    }
    
    fn read(&self) -> ScreenChar {
        self.inner
    }
    
    fn write(&mut self, ch: ScreenChar) {
        self.inner = ch;
    }
}

#[repr(transparent)]
struct Buffer {
    chars: [[VolatileScreenChar; BUFFER_WIDTH]; BUFFER_HEIGHT],
}

pub struct Writer {
    pub column_position: usize,
    pub color_code: ColorCode,
    buffer: &'static mut Buffer,
}

impl Writer {
    pub fn write_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.new_line(),
            byte => {
                if self.column_position >= BUFFER_WIDTH {
                    self.new_line();
                }

                let row = BUFFER_HEIGHT - 1;
                let col = self.column_position;

                let color_code = self.color_code;
                self.buffer.chars[row][col].write(ScreenChar {
                    ascii_character: byte,
                    color_code,
                });
                self.column_position += 1;
            }
        }
    }

    pub fn write_string(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                // printable ASCII byte or newline
                0x20..=0x7e | b'\n' => self.write_byte(byte),
                // not part of printable ASCII range
                _ => self.write_byte(0xfe),
            }
        }
    }

    fn new_line(&mut self) {
        for row in 1..BUFFER_HEIGHT {
            for col in 0..BUFFER_WIDTH {
                let character = self.buffer.chars[row][col].read();
                self.buffer.chars[row - 1][col].write(character);
            }
        }
        self.clear_row(BUFFER_HEIGHT - 1);
        self.column_position = 0;
    }

    fn clear_row(&mut self, row: usize) {
        let blank = ScreenChar {
            ascii_character: b' ',
            color_code: self.color_code,
        };
        for col in 0..BUFFER_WIDTH {
            self.buffer.chars[row][col].write(blank);
        }
    }
    
    pub fn clear_screen(&mut self) {
        for row in 0..BUFFER_HEIGHT {
            self.clear_row(row);
        }
        self.column_position = 0;
    }
    
    pub fn set_position(&mut self, row: usize, col: usize) {
        if row < BUFFER_HEIGHT && col < BUFFER_WIDTH {
            // We can't actually move the cursor in VGA text mode without more code,
            // but we can update our internal position for the next write
            self.column_position = col;
            // The row position is implicit in our implementation
        }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_string(s);
        Ok(())
    }
}

// Initialize the VGA buffer with empty characters
fn init_buffer() -> Buffer {
    let mut buffer: Buffer = unsafe { core::mem::zeroed() };
    let blank = ScreenChar {
        ascii_character: b' ',
        color_code: ColorCode::new(Color::White, Color::Black),
    };
    
    for row in 0..BUFFER_HEIGHT {
        for col in 0..BUFFER_WIDTH {
            buffer.chars[row][col] = VolatileScreenChar::new(blank);
        }
    }
    
    buffer
}

lazy_static! {
    pub static ref WRITER: Mutex<Writer> = {
        let buffer = unsafe {
            // Create a static buffer
            static mut BUFFER: Buffer = unsafe { core::mem::zeroed() };
            &mut BUFFER
        };
        
        // Initialize the buffer with empty characters
        let blank = ScreenChar {
            ascii_character: b' ',
            color_code: ColorCode::new(Color::White, Color::Black),
        };
        
        for row in 0..BUFFER_HEIGHT {
            for col in 0..BUFFER_WIDTH {
                buffer.chars[row][col] = VolatileScreenChar::new(blank);
            }
        }
        
        Mutex::new(Writer {
            column_position: 0,
            color_code: ColorCode::new(Color::LightGreen, Color::Black),
            buffer,
        })
    };
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::vga_buffer::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    use x86_64::instructions::interrupts;

    interrupts::without_interrupts(|| {
        WRITER.lock().write_fmt(args).unwrap();
    });
}
