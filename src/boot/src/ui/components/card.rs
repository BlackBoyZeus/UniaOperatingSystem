// Card component for UNIA OS UI
use alloc::string::String;
use alloc::vec::Vec;

pub struct Card {
    title: String,
    content: Vec<String>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Card {
    pub fn new(title: String, x: usize, y: usize, width: usize, height: usize) -> Self {
        Card {
            title,
            content: Vec::new(),
            x,
            y,
            width,
            height,
        }
    }

    pub fn add_content(&mut self, line: String) {
        self.content.push(line);
    }

    pub fn get_title(&self) -> &str {
        &self.title
    }

    pub fn get_content(&self) -> &[String] {
        &self.content
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    pub fn get_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}
