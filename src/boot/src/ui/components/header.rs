// Header component for UNIA OS UI
use alloc::string::String;

pub struct Header {
    title: String,
    subtitle: Option<String>,
    x: usize,
    y: usize,
    width: usize,
}

impl Header {
    pub fn new(title: String, x: usize, y: usize, width: usize) -> Self {
        Header {
            title,
            subtitle: None,
            x,
            y,
            width,
        }
    }

    pub fn with_subtitle(mut self, subtitle: String) -> Self {
        self.subtitle = Some(subtitle);
        self
    }

    pub fn get_title(&self) -> &str {
        &self.title
    }

    pub fn get_subtitle(&self) -> Option<&str> {
        self.subtitle.as_deref()
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    pub fn get_width(&self) -> usize {
        self.width
    }
}
