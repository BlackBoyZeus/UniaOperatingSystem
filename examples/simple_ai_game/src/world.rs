use std::time::Duration;

/// Time of day in the game world
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeOfDay {
    /// Morning (6:00 - 10:00)
    Morning,
    
    /// Day (10:00 - 18:00)
    Day,
    
    /// Evening (18:00 - 22:00)
    Evening,
    
    /// Night (22:00 - 6:00)
    Night,
}

/// Terrain type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerrainType {
    /// Grass
    Grass,
    
    /// Water
    Water,
    
    /// Forest
    Forest,
    
    /// Mountain
    Mountain,
    
    /// Road
    Road,
    
    /// Sand
    Sand,
}

/// Terrain tile
pub struct TerrainTile {
    /// Terrain type
    pub terrain_type: TerrainType,
    
    /// Elevation (0.0 - 1.0)
    pub elevation: f32,
    
    /// Moisture (0.0 - 1.0)
    pub moisture: f32,
    
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
}

/// Game world
pub struct World {
    /// World width
    pub width: usize,
    
    /// World height
    pub height: usize,
    
    /// Terrain data
    pub terrain: Vec<Vec<TerrainTile>>,
    
    /// Current time of day
    pub time_of_day: TimeOfDay,
    
    /// Game time in seconds
    pub game_time: f32,
    
    /// Day length in seconds
    pub day_length: f32,
}

impl World {
    /// Create a new world
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            terrain: Vec::new(),
            time_of_day: TimeOfDay::Morning,
            game_time: 0.0,
            day_length: 600.0, // 10 minutes per day
        }
    }
    
    /// Generate terrain
    pub fn generate_terrain(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        
        // Create terrain grid
        let mut terrain = Vec::with_capacity(height);
        
        // Use simplex noise for terrain generation
        let seed = 12345; // Fixed seed for reproducibility
        let noise = noise::OpenSimplex::new(seed);
        
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            
            for x in 0..width {
                // Generate noise values
                let nx = x as f64 / width as f64;
                let ny = y as f64 / height as f64;
                
                // Use different frequencies for different features
                let elevation = (noise.get([nx * 5.0, ny * 5.0]) as f32 + 1.0) / 2.0;
                let moisture = (noise.get([nx * 10.0, ny * 10.0, 0.5]) as f32 + 1.0) / 2.0;
                let temperature = (noise.get([nx * 3.0, ny * 3.0, 1.0]) as f32 + 1.0) / 2.0;
                
                // Determine terrain type based on elevation and moisture
                let terrain_type = if elevation < 0.3 {
                    TerrainType::Water
                } else if elevation > 0.8 {
                    TerrainType::Mountain
                } else if moisture > 0.7 && elevation < 0.7 {
                    TerrainType::Forest
                } else if moisture < 0.3 && temperature > 0.7 {
                    TerrainType::Sand
                } else {
                    TerrainType::Grass
                };
                
                // Create terrain tile
                let tile = TerrainTile {
                    terrain_type,
                    elevation,
                    moisture,
                    temperature,
                };
                
                row.push(tile);
            }
            
            terrain.push(row);
        }
        
        // Add some roads
        self.add_roads(&mut terrain);
        
        self.terrain = terrain;
    }
    
    /// Add roads to the terrain
    fn add_roads(&self, terrain: &mut Vec<Vec<TerrainTile>>) {
        // Add a horizontal road
        let road_y = self.height / 2;
        for x in 0..self.width {
            if terrain[road_y][x].terrain_type != TerrainType::Water {
                terrain[road_y][x].terrain_type = TerrainType::Road;
            }
        }
        
        // Add a vertical road
        let road_x = self.width / 2;
        for y in 0..self.height {
            if terrain[y][road_x].terrain_type != TerrainType::Water {
                terrain[y][road_x].terrain_type = TerrainType::Road;
            }
        }
    }
    
    /// Update the world
    pub fn update(&mut self, delta_time: f32) {
        // Update game time
        self.game_time += delta_time;
        
        // Wrap game time to day length
        while self.game_time >= self.day_length {
            self.game_time -= self.day_length;
        }
        
        // Update time of day
        self.update_time_of_day();
    }
    
    /// Update time of day
    pub fn update_time_of_day(&mut self) {
        // Calculate time of day
        let day_progress = self.game_time / self.day_length;
        
        // Update time of day
        self.time_of_day = if day_progress < 0.25 {
            TimeOfDay::Morning
        } else if day_progress < 0.5 {
            TimeOfDay::Day
        } else if day_progress < 0.75 {
            TimeOfDay::Evening
        } else {
            TimeOfDay::Night
        };
    }
    
    /// Get terrain at position
    pub fn get_terrain_at(&self, x: usize, y: usize) -> Option<&TerrainTile> {
        if x < self.width && y < self.height {
            Some(&self.terrain[y][x])
        } else {
            None
        }
    }
    
    /// Check if position is walkable
    pub fn is_walkable(&self, x: usize, y: usize) -> bool {
        if let Some(tile) = self.get_terrain_at(x, y) {
            match tile.terrain_type {
                TerrainType::Water | TerrainType::Mountain => false,
                _ => true,
            }
        } else {
            false
        }
    }
    
    /// Get movement cost for terrain
    pub fn get_movement_cost(&self, x: usize, y: usize) -> f32 {
        if let Some(tile) = self.get_terrain_at(x, y) {
            match tile.terrain_type {
                TerrainType::Road => 0.8,
                TerrainType::Grass => 1.0,
                TerrainType::Forest => 1.5,
                TerrainType::Sand => 1.2,
                TerrainType::Water => 5.0,
                TerrainType::Mountain => 3.0,
            }
        } else {
            1.0
        }
    }
}
