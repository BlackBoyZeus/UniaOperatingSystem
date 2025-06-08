/// Item in the inventory
pub struct Item {
    /// Item name
    pub name: String,
    
    /// Item value
    pub value: i32,
    
    /// Item quantity
    pub quantity: i32,
    
    /// Item description
    pub description: Option<String>,
    
    /// Item properties (key-value pairs)
    pub properties: std::collections::HashMap<String, String>,
}

impl Item {
    /// Create a new item
    pub fn new(name: &str, value: i32) -> Self {
        Self {
            name: name.to_string(),
            value,
            quantity: 1,
            description: None,
            properties: std::collections::HashMap::new(),
        }
    }
    
    /// Create a new item with description
    pub fn with_description(name: &str, value: i32, description: &str) -> Self {
        let mut item = Self::new(name, value);
        item.description = Some(description.to_string());
        item
    }
    
    /// Add a property to the item
    pub fn add_property(&mut self, key: &str, value: &str) {
        self.properties.insert(key.to_string(), value.to_string());
    }
    
    /// Get a property from the item
    pub fn get_property(&self, key: &str) -> Option<&String> {
        self.properties.get(key)
    }
}

/// Inventory system
pub struct Inventory {
    /// Items in the inventory
    items: Vec<Item>,
    
    /// Maximum number of items
    max_items: usize,
    
    /// Total weight
    total_weight: f32,
    
    /// Maximum weight
    max_weight: f32,
}

impl Inventory {
    /// Create a new inventory
    pub fn new(max_items: usize) -> Self {
        Self {
            items: Vec::new(),
            max_items,
            total_weight: 0.0,
            max_weight: 100.0,
        }
    }
    
    /// Add an item to the inventory
    pub fn add_item(&mut self, item: Item) -> bool {
        // Check if inventory is full
        if self.items.len() >= self.max_items {
            println!("Inventory is full!");
            return false;
        }
        
        // Check if we already have this item (for stackable items)
        for existing_item in &mut self.items {
            if existing_item.name == item.name {
                // Stack items
                existing_item.quantity += item.quantity;
                return true;
            }
        }
        
        // Add new item
        self.items.push(item);
        true
    }
    
    /// Remove an item from the inventory
    pub fn remove_item(&mut self, name: &str, quantity: i32) -> bool {
        for i in 0..self.items.len() {
            if self.items[i].name == name {
                if self.items[i].quantity <= quantity {
                    // Remove the entire stack
                    self.items.remove(i);
                } else {
                    // Remove part of the stack
                    self.items[i].quantity -= quantity;
                }
                return true;
            }
        }
        
        false
    }
    
    /// Get an item from the inventory
    pub fn get_item(&self, name: &str) -> Option<&Item> {
        self.items.iter().find(|item| item.name == name)
    }
    
    /// Get a mutable reference to an item
    pub fn get_item_mut(&mut self, name: &str) -> Option<&mut Item> {
        self.items.iter_mut().find(|item| item.name == name)
    }
    
    /// Get all items in the inventory
    pub fn get_items(&self) -> &[Item] {
        &self.items
    }
    
    /// Get the number of items in the inventory
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
    
    /// Check if the inventory is full
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.max_items
    }
    
    /// Get the total value of all items
    pub fn total_value(&self) -> i32 {
        self.items.iter().map(|item| item.value * item.quantity).sum()
    }
    
    /// Clear the inventory
    pub fn clear(&mut self) {
        self.items.clear();
    }
}
