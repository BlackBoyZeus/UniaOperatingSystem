use embedded_graphics::{
    pixelcolor::Rgb888,
    prelude::*,
    primitives::{Circle, Line, PrimitiveStyle, Rectangle, Triangle},
};
use spin::Mutex;

mod display;
mod font;
mod framebuffer;
mod window;

pub use display::{Display, DisplayMode};
pub use framebuffer::Framebuffer;
pub use window::{Window, WindowManager};

/// Modern graphics system for UNIA OS
pub struct GraphicsSystem {
    display: Mutex<Display>,
    window_manager: Mutex<WindowManager>,
    framebuffer: Mutex<Framebuffer>,
}

impl GraphicsSystem {
    pub fn new() -> Self {
        Self {
            display: Mutex::new(Display::new()),
            window_manager: Mutex::new(WindowManager::new()),
            framebuffer: Mutex::new(Framebuffer::new()),
        }
    }

    pub fn init(&mut self) {
        // Initialize display
        self.display.lock().init();

        // Initialize window manager
        self.window_manager.lock().init();

        // Initialize framebuffer
        self.framebuffer.lock().init();

        // Set up default display mode
        self.set_display_mode(DisplayMode::Graphics {
            width: 1024,
            height: 768,
            bpp: 32,
        });
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.display.lock().set_mode(mode);
        self.framebuffer.lock().resize(mode.dimensions());
        self.window_manager.lock().update_resolution(mode.dimensions());
    }

    pub fn draw_pixel(&mut self, x: i32, y: i32, color: Color) {
        if let Some(fb) = self.framebuffer.lock().as_mut() {
            fb.draw_pixel(x, y, color);
        }
    }

    pub fn draw_line(&mut self, start: Point, end: Point, color: Color) {
        Line::new(start, end)
            .into_styled(PrimitiveStyle::with_stroke(color, 1))
            .draw(&mut *self.framebuffer.lock())
            .ok();
    }

    pub fn draw_rectangle(&mut self, top_left: Point, size: Size, color: Color) {
        Rectangle::new(top_left, size)
            .into_styled(PrimitiveStyle::with_stroke(color, 1))
            .draw(&mut *self.framebuffer.lock())
            .ok();
    }

    pub fn draw_circle(&mut self, center: Point, radius: u32, color: Color) {
        Circle::new(center, radius)
            .into_styled(PrimitiveStyle::with_stroke(color, 1))
            .draw(&mut *self.framebuffer.lock())
            .ok();
    }

    pub fn draw_triangle(&mut self, p1: Point, p2: Point, p3: Point, color: Color) {
        Triangle::new(p1, p2, p3)
            .into_styled(PrimitiveStyle::with_stroke(color, 1))
            .draw(&mut *self.framebuffer.lock())
            .ok();
    }

    pub fn draw_text(&mut self, text: &str, position: Point, color: Color) {
        font::draw_text(&mut *self.framebuffer.lock(), text, position, color);
    }

    pub fn clear_screen(&mut self, color: Color) {
        self.framebuffer.lock().clear(color);
    }

    pub fn swap_buffers(&mut self) {
        self.framebuffer.lock().swap_buffers();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black = 0x000000,
    White = 0xFFFFFF,
    Red = 0xFF0000,
    Green = 0x00FF00,
    Blue = 0x0000FF,
    Yellow = 0xFFFF00,
    Magenta = 0xFF00FF,
    Cyan = 0x00FFFF,
    Gray = 0x808080,
}

impl From<Color> for Rgb888 {
    fn from(color: Color) -> Self {
        let rgb = color as u32;
        Rgb888::new(
            ((rgb >> 16) & 0xFF) as u8,
            ((rgb >> 8) & 0xFF) as u8,
            (rgb & 0xFF) as u8,
        )
    }
}

/// Graphics configuration
#[derive(Debug, Clone)]
pub struct GraphicsConfig {
    pub mode: DisplayMode,
    pub vsync: bool,
    pub double_buffer: bool,
    pub hardware_cursor: bool,
    pub acceleration: AccelerationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationMode {
    None,
    Basic2D,
    Full2D,
    Basic3D,
    Full3D,
}

/// Window system theme
#[derive(Debug, Clone)]
pub struct Theme {
    pub window_background: Color,
    pub window_border: Color,
    pub title_bar_active: Color,
    pub title_bar_inactive: Color,
    pub text_color: Color,
    pub button_color: Color,
    pub button_hover: Color,
    pub button_press: Color,
}
