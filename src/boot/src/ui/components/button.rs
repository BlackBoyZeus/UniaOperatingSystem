// Basic button component for UNIA OS UI
use alloc::string::String;

pub struct Button {
    text: String,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    is_active: bool,
}

impl Button {
    pub fn new(text: String, x: usize, y: usize, width: usize, height: usize) -> Self {
        Button {
            text,
            x,
            y,
            width,
            height,
            is_active: false,
        }
    }

    pub fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }

    pub fn get_text(&self) -> &str {
        &self.text
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    pub fn get_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}
