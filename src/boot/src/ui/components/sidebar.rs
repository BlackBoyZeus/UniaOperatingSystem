// Sidebar component for UNIA OS UI
use alloc::string::String;
use alloc::vec::Vec;

pub struct SidebarItem {
    text: String,
    is_active: bool,
}

impl SidebarItem {
    pub fn new(text: String) -> Self {
        SidebarItem {
            text,
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
}

pub struct Sidebar {
    title: String,
    items: Vec<SidebarItem>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Sidebar {
    pub fn new(title: String, x: usize, y: usize, width: usize, height: usize) -> Self {
        Sidebar {
            title,
            items: Vec::new(),
            x,
            y,
            width,
            height,
        }
    }

    pub fn add_item(&mut self, text: String) {
        self.items.push(SidebarItem::new(text));
    }

    pub fn set_active_item(&mut self, index: usize) {
        if index < self.items.len() {
            for (i, item) in self.items.iter_mut().enumerate() {
                item.set_active(i == index);
            }
        }
    }

    pub fn get_title(&self) -> &str {
        &self.title
    }

    pub fn get_items(&self) -> &[SidebarItem] {
        &self.items
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    pub fn get_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}
