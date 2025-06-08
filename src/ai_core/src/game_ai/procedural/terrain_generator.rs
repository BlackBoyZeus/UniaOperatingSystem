//! Terrain generator for procedural world creation.
//!
//! This module provides advanced terrain generation capabilities using
//! noise functions, erosion simulation, and AI-driven feature placement.

use std::collections::HashMap;
use std::sync::Arc;

use noise::{NoiseFn, Perlin, Fbm, Turbulence, Worley, SuperSimplex};
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::prelude::ThreadRng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::error::{AIError, Result};
use crate::inference::{InferenceEngine, InferenceOptions, Tensor};

/// Terrain type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainType {
    /// Ocean
    Ocean,
    /// Beach
    Beach,
    /// Plains
    Plains,
    /// Forest
    Forest,
    /// Hills
    Hills,
    /// Mountains
    Mountains,
    /// Snow
    Snow,
    /// Desert
    Desert,
    /// Swamp
    Swamp,
    /// Jungle
    Jungle,
    /// Tundra
    Tundra,
    /// Savanna
    Savanna,
    /// Volcano
    Volcano,
    /// Canyon
    Canyon,
    /// Crater
    Crater,
}

/// Biome definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Biome {
    /// Biome name
    pub name: String,
    
    /// Terrain type
    pub terrain_type: TerrainType,
    
    /// Temperature range (0.0 - 1.0)
    pub temperature_range: (f32, f32),
    
    /// Moisture range (0.0 - 1.0)
    pub moisture_range: (f32, f32),
    
    /// Height range (0.0 - 1.0)
    pub height_range: (f32, f32),
    
    /// Color (RGBA)
    pub color: [u8; 4],
    
    /// Features that can appear in this biome
    pub features: Vec<String>,
}

/// Terrain feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainFeature {
    /// Feature type
    pub feature_type: String,
    
    /// Position
    pub position: [f32; 2],
    
    /// Rotation
    pub rotation: f32,
    
    /// Scale
    pub scale: f32,
    
    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Terrain heightmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heightmap {
    /// Width
    pub width: usize,
    
    /// Height
    pub height: usize,
    
    /// Data
    pub data: Vec<f32>,
}

impl Heightmap {
    /// Create a new heightmap.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }
    
    /// Get the height at a specific position.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }
    
    /// Set the height at a specific position.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }
    
    /// Get the normalized gradient at a specific position.
    pub fn gradient(&self, x: usize, y: usize) -> [f32; 2] {
        let mut gradient = [0.0, 0.0];
        
        if x > 0 && x < self.width - 1 && y > 0 && y < self.height - 1 {
            // X gradient (Sobel operator)
            gradient[0] = (
                -1.0 * self.get(x - 1, y - 1) +
                -2.0 * self.get(x - 1, y) +
                -1.0 * self.get(x - 1, y + 1) +
                1.0 * self.get(x + 1, y - 1) +
                2.0 * self.get(x + 1, y) +
                1.0 * self.get(x + 1, y + 1)
            ) / 8.0;
            
            // Y gradient (Sobel operator)
            gradient[1] = (
                -1.0 * self.get(x - 1, y - 1) +
                -2.0 * self.get(x, y - 1) +
                -1.0 * self.get(x + 1, y - 1) +
                1.0 * self.get(x - 1, y + 1) +
                2.0 * self.get(x, y + 1) +
                1.0 * self.get(x + 1, y + 1)
            ) / 8.0;
        }
        
        gradient
    }
    
    /// Get the slope at a specific position.
    pub fn slope(&self, x: usize, y: usize) -> f32 {
        let gradient = self.gradient(x, y);
        (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt()
    }
    
    /// Normalize the heightmap to the range [0, 1].
    pub fn normalize(&mut self) {
        let mut min_height = f32::MAX;
        let mut max_height = f32::MIN;
        
        // Find min and max heights
        for &height in &self.data {
            min_height = min_height.min(height);
            max_height = max_height.max(height);
        }
        
        let range = max_height - min_height;
        
        // Normalize heights
        if range > 0.0 {
            for height in &mut self.data {
                *height = (*height - min_height) / range;
            }
        }
    }
    
    /// Apply erosion to the heightmap.
    pub fn apply_erosion(&mut self, iterations: usize, strength: f32) {
        let mut rng = rand::thread_rng();
        
        for _ in 0..iterations {
            // Start a droplet at a random position
            let mut x = rng.gen_range(0..self.width) as f32;
            let mut y = rng.gen_range(0..self.height) as f32;
            let mut speed = 0.0;
            let mut water = 1.0;
            let mut sediment = 0.0;
            
            // Droplet lifetime
            for _ in 0..100 {
                // Get current cell indices
                let cell_x = x.floor() as usize;
                let cell_y = y.floor() as usize;
                
                if cell_x >= self.width - 1 || cell_y >= self.height - 1 {
                    break;
                }
                
                // Calculate droplet's offset inside the cell (0,0) to (1,1)
                let cell_offset_x = x - cell_x as f32;
                let cell_offset_y = y - cell_y as f32;
                
                // Calculate heights at the corners of the cell
                let height_nw = self.get(cell_x, cell_y);
                let height_ne = self.get(cell_x + 1, cell_y);
                let height_sw = self.get(cell_x, cell_y + 1);
                let height_se = self.get(cell_x + 1, cell_y + 1);
                
                // Calculate droplet's direction (gradient of the height field)
                let gradient_x = (height_ne - height_nw) * (1.0 - cell_offset_y) + 
                                 (height_se - height_sw) * cell_offset_y;
                let gradient_y = (height_sw - height_nw) * (1.0 - cell_offset_x) + 
                                 (height_se - height_ne) * cell_offset_x;
                
                // Update droplet's direction and position
                let dir_x = gradient_x * 0.1 - 0.05;
                let dir_y = gradient_y * 0.1 - 0.05;
                
                x += dir_x;
                y += dir_y;
                
                // Stop if droplet has flowed off the map
                if x < 0.0 || x >= self.width as f32 - 1.0 || y < 0.0 || y >= self.height as f32 - 1.0 {
                    break;
                }
                
                // Calculate new cell indices
                let new_cell_x = x.floor() as usize;
                let new_cell_y = y.floor() as usize;
                
                // Calculate droplet's height and delta height
                let old_height = height_nw * (1.0 - cell_offset_x) * (1.0 - cell_offset_y) +
                                height_ne * cell_offset_x * (1.0 - cell_offset_y) +
                                height_sw * (1.0 - cell_offset_x) * cell_offset_y +
                                height_se * cell_offset_x * cell_offset_y;
                
                let new_height = self.get(new_cell_x, new_cell_y);
                let delta_height = new_height - old_height;
                
                // Calculate droplet's sediment capacity
                let capacity = (-delta_height).max(0.0) * speed * water * strength;
                
                // If carrying more sediment than capacity, deposit some
                if sediment > capacity {
                    let amount_to_deposit = (sediment - capacity) * 0.1;
                    sediment -= amount_to_deposit;
                    
                    // Deposit sediment at the current cell
                    self.set(cell_x, cell_y, self.get(cell_x, cell_y) + amount_to_deposit);
                }
                // If carrying less sediment than capacity, erode some
                else if delta_height > 0.0 {
                    let amount_to_erode = (capacity - sediment) * 0.1;
                    sediment += amount_to_erode;
                    
                    // Erode from the current cell
                    self.set(cell_x, cell_y, self.get(cell_x, cell_y) - amount_to_erode);
                }
                
                // Update speed and water
                speed = (speed * 0.9 + delta_height * 0.1).max(0.0);
                water *= 0.99;
            }
        }
    }
}

/// Terrain generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainGenerationParams {
    /// Width of the terrain
    pub width: usize,
    
    /// Height of the terrain
    pub height: usize,
    
    /// Seed for random generation
    pub seed: u64,
    
    /// Scale of the terrain
    pub scale: f32,
    
    /// Octaves for noise generation
    pub octaves: usize,
    
    /// Persistence for noise generation
    pub persistence: f32,
    
    /// Lacunarity for noise generation
    pub lacunarity: f32,
    
    /// Erosion iterations
    pub erosion_iterations: usize,
    
    /// Erosion strength
    pub erosion_strength: f32,
    
    /// Sea level (0.0 - 1.0)
    pub sea_level: f32,
    
    /// Mountain level (0.0 - 1.0)
    pub mountain_level: f32,
    
    /// Temperature gradient (north-south)
    pub temperature_gradient: f32,
    
    /// Moisture gradient (east-west)
    pub moisture_gradient: f32,
    
    /// Biome definitions
    pub biomes: Vec<Biome>,
    
    /// Feature placement density
    pub feature_density: f32,
}

impl Default for TerrainGenerationParams {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            seed: 42,
            scale: 100.0,
            octaves: 6,
            persistence: 0.5,
            lacunarity: 2.0,
            erosion_iterations: 1000,
            erosion_strength: 0.3,
            sea_level: 0.4,
            mountain_level: 0.7,
            temperature_gradient: 1.0,
            moisture_gradient: 1.0,
            biomes: vec![
                Biome {
                    name: "Ocean".to_string(),
                    terrain_type: TerrainType::Ocean,
                    temperature_range: (0.0, 1.0),
                    moisture_range: (0.0, 1.0),
                    height_range: (0.0, 0.4),
                    color: [0, 0, 200, 255],
                    features: vec!["coral_reef".to_string(), "shipwreck".to_string()],
                },
                Biome {
                    name: "Beach".to_string(),
                    terrain_type: TerrainType::Beach,
                    temperature_range: (0.3, 1.0),
                    moisture_range: (0.0, 0.6),
                    height_range: (0.4, 0.45),
                    color: [240, 240, 160, 255],
                    features: vec!["palm_tree".to_string(), "rocks".to_string()],
                },
                Biome {
                    name: "Plains".to_string(),
                    terrain_type: TerrainType::Plains,
                    temperature_range: (0.3, 0.7),
                    moisture_range: (0.3, 0.6),
                    height_range: (0.45, 0.55),
                    color: [100, 220, 100, 255],
                    features: vec!["grass".to_string(), "flowers".to_string(), "village".to_string()],
                },
                Biome {
                    name: "Forest".to_string(),
                    terrain_type: TerrainType::Forest,
                    temperature_range: (0.3, 0.7),
                    moisture_range: (0.6, 1.0),
                    height_range: (0.45, 0.65),
                    color: [0, 160, 0, 255],
                    features: vec!["trees".to_string(), "mushrooms".to_string(), "cabin".to_string()],
                },
                Biome {
                    name: "Hills".to_string(),
                    terrain_type: TerrainType::Hills,
                    temperature_range: (0.2, 0.7),
                    moisture_range: (0.3, 0.8),
                    height_range: (0.55, 0.7),
                    color: [160, 160, 100, 255],
                    features: vec!["rocks".to_string(), "cave".to_string()],
                },
                Biome {
                    name: "Mountains".to_string(),
                    terrain_type: TerrainType::Mountains,
                    temperature_range: (0.0, 0.5),
                    moisture_range: (0.2, 0.7),
                    height_range: (0.7, 0.85),
                    color: [120, 120, 120, 255],
                    features: vec!["peak".to_string(), "cave".to_string(), "mine".to_string()],
                },
                Biome {
                    name: "Snow".to_string(),
                    terrain_type: TerrainType::Snow,
                    temperature_range: (0.0, 0.3),
                    moisture_range: (0.4, 1.0),
                    height_range: (0.65, 1.0),
                    color: [255, 255, 255, 255],
                    features: vec!["ice_spikes".to_string(), "frozen_lake".to_string()],
                },
                Biome {
                    name: "Desert".to_string(),
                    terrain_type: TerrainType::Desert,
                    temperature_range: (0.7, 1.0),
                    moisture_range: (0.0, 0.3),
                    height_range: (0.45, 0.6),
                    color: [240, 240, 160, 255],
                    features: vec!["cactus".to_string(), "oasis".to_string(), "pyramid".to_string()],
                },
                Biome {
                    name: "Swamp".to_string(),
                    terrain_type: TerrainType::Swamp,
                    temperature_range: (0.5, 0.8),
                    moisture_range: (0.7, 1.0),
                    height_range: (0.45, 0.5),
                    color: [70, 100, 70, 255],
                    features: vec!["swamp_tree".to_string(), "mushrooms".to_string(), "hut".to_string()],
                },
            ],
            feature_density: 0.01,
        }
    }
}

/// Generated terrain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terrain {
    /// Width of the terrain
    pub width: usize,
    
    /// Height of the terrain
    pub height: usize,
    
    /// Heightmap
    pub heightmap: Heightmap,
    
    /// Biome map
    pub biome_map: Vec<TerrainType>,
    
    /// Temperature map
    pub temperature_map: Vec<f32>,
    
    /// Moisture map
    pub moisture_map: Vec<f32>,
    
    /// Features
    pub features: Vec<TerrainFeature>,
}

/// Terrain generator.
pub struct TerrainGenerator {
    /// Inference engine for AI-driven feature placement
    inference_engine: Option<Arc<InferenceEngine>>,
    
    /// Default parameters
    default_params: TerrainGenerationParams,
}

impl TerrainGenerator {
    /// Create a new terrain generator.
    pub fn new(inference_engine: Option<Arc<InferenceEngine>>) -> Self {
        Self {
            inference_engine,
            default_params: TerrainGenerationParams::default(),
        }
    }
    
    /// Generate terrain.
    pub async fn generate(&self, params: Option<TerrainGenerationParams>) -> Result<Terrain> {
        let params = params.unwrap_or_else(|| self.default_params.clone());
        
        // Create RNG
        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        
        // Generate heightmap
        let mut heightmap = self.generate_heightmap(&params, &mut rng);
        
        // Apply erosion
        if params.erosion_iterations > 0 {
            heightmap.apply_erosion(params.erosion_iterations, params.erosion_strength);
            heightmap.normalize();
        }
        
        // Generate temperature and moisture maps
        let temperature_map = self.generate_temperature_map(&params, &heightmap, &mut rng);
        let moisture_map = self.generate_moisture_map(&params, &heightmap, &mut rng);
        
        // Generate biome map
        let biome_map = self.generate_biome_map(&params, &heightmap, &temperature_map, &moisture_map);
        
        // Generate features
        let features = self.generate_features(&params, &heightmap, &biome_map, &mut rng).await?;
        
        Ok(Terrain {
            width: params.width,
            height: params.height,
            heightmap,
            biome_map,
            temperature_map,
            moisture_map,
            features,
        })
    }
    
    /// Generate heightmap.
    fn generate_heightmap(&self, params: &TerrainGenerationParams, rng: &mut ChaCha8Rng) -> Heightmap {
        let mut heightmap = Heightmap::new(params.width, params.height);
        
        // Create noise function
        let noise_seed = rng.gen::<u32>();
        let noise = Fbm::<Perlin>::new(noise_seed)
            .set_octaves(params.octaves as usize)
            .set_persistence(params.persistence)
            .set_lacunarity(params.lacunarity);
        
        // Generate base terrain
        for y in 0..params.height {
            for x in 0..params.width {
                let nx = x as f64 / params.width as f64;
                let ny = y as f64 / params.height as f64;
                
                // Sample noise
                let value = noise.get([
                    nx * params.scale as f64,
                    ny * params.scale as f64,
                ]) as f32;
                
                // Normalize to [0, 1]
                let normalized = (value + 1.0) * 0.5;
                
                // Apply island mask (higher at center, lower at edges)
                let dx = nx - 0.5;
                let dy = ny - 0.5;
                let distance = (dx * dx + dy * dy).sqrt();
                let island_mask = 1.0 - (distance * 2.0).min(1.0).powf(2.0);
                
                // Combine noise with island mask
                let height = normalized * island_mask;
                
                heightmap.set(x, y, height);
            }
        }
        
        // Normalize heightmap
        heightmap.normalize();
        
        heightmap
    }
    
    /// Generate temperature map.
    fn generate_temperature_map(
        &self,
        params: &TerrainGenerationParams,
        heightmap: &Heightmap,
        rng: &mut ChaCha8Rng,
    ) -> Vec<f32> {
        let mut temperature_map = vec![0.0; params.width * params.height];
        
        // Create noise function for temperature variations
        let noise_seed = rng.gen::<u32>();
        let noise = Fbm::<Perlin>::new(noise_seed)
            .set_octaves(4)
            .set_persistence(0.5)
            .set_lacunarity(2.0);
        
        // Generate temperature map
        for y in 0..params.height {
            for x in 0..params.width {
                let nx = x as f64 / params.width as f64;
                let ny = y as f64 / params.height as f64;
                
                // Base temperature depends on latitude (y-coordinate)
                // Higher latitudes (top and bottom) are colder
                let latitude_factor = 1.0 - (ny - 0.5).abs() * 2.0;
                let base_temperature = latitude_factor.powf(params.temperature_gradient as f64) as f32;
                
                // Temperature decreases with altitude
                let height = heightmap.get(x, y);
                let altitude_factor = 1.0 - (height - params.sea_level).max(0.0) * 0.5;
                
                // Add some noise for local variations
                let noise_value = noise.get([
                    nx * 3.0,
                    ny * 3.0,
                ]) as f32;
                let noise_factor = (noise_value + 1.0) * 0.1; // ±10% variation
                
                // Combine factors
                let temperature = (base_temperature * altitude_factor + noise_factor).clamp(0.0, 1.0);
                
                temperature_map[y * params.width + x] = temperature;
            }
        }
        
        temperature_map
    }
    
    /// Generate moisture map.
    fn generate_moisture_map(
        &self,
        params: &TerrainGenerationParams,
        heightmap: &Heightmap,
        rng: &mut ChaCha8Rng,
    ) -> Vec<f32> {
        let mut moisture_map = vec![0.0; params.width * params.height];
        
        // Create noise function for moisture variations
        let noise_seed = rng.gen::<u32>();
        let noise = Fbm::<Perlin>::new(noise_seed)
            .set_octaves(4)
            .set_persistence(0.5)
            .set_lacunarity(2.0);
        
        // Generate moisture map
        for y in 0..params.height {
            for x in 0..params.width {
                let nx = x as f64 / params.width as f64;
                let ny = y as f64 / params.height as f64;
                
                // Base moisture depends on longitude (x-coordinate) and latitude
                // This creates a simple climate model
                let longitude_factor = (nx - 0.5).abs() * 2.0;
                let latitude_factor = (ny - 0.5).abs() * 2.0;
                let base_moisture = (1.0 - longitude_factor.powf(params.moisture_gradient as f64)) as f32;
                
                // Moisture is higher near water and lower at high altitudes
                let height = heightmap.get(x, y);
                let water_factor = if height < params.sea_level {
                    1.0
                } else {
                    (1.0 - (height - params.sea_level) * 2.0).max(0.0)
                };
                
                // Add some noise for local variations
                let noise_value = noise.get([
                    nx * 4.0,
                    ny * 4.0,
                ]) as f32;
                let noise_factor = (noise_value + 1.0) * 0.15; // ±15% variation
                
                // Combine factors
                let moisture = (base_moisture * 0.5 + water_factor * 0.4 + noise_factor).clamp(0.0, 1.0);
                
                moisture_map[y * params.width + x] = moisture;
            }
        }
        
        moisture_map
    }
    
    /// Generate biome map.
    fn generate_biome_map(
        &self,
        params: &TerrainGenerationParams,
        heightmap: &Heightmap,
        temperature_map: &[f32],
        moisture_map: &[f32],
    ) -> Vec<TerrainType> {
        let mut biome_map = vec![TerrainType::Ocean; params.width * params.height];
        
        // Assign biomes based on height, temperature, and moisture
        for y in 0..params.height {
            for x in 0..params.width {
                let index = y * params.width + x;
                let height = heightmap.get(x, y);
                let temperature = temperature_map[index];
                let moisture = moisture_map[index];
                
                // Find matching biome
                for biome in &params.biomes {
                    if height >= biome.height_range.0 && height < biome.height_range.1 &&
                       temperature >= biome.temperature_range.0 && temperature < biome.temperature_range.1 &&
                       moisture >= biome.moisture_range.0 && moisture < biome.moisture_range.1 {
                        biome_map[index] = biome.terrain_type;
                        break;
                    }
                }
            }
        }
        
        biome_map
    }
    
    /// Generate terrain features.
    async fn generate_features(
        &self,
        params: &TerrainGenerationParams,
        heightmap: &Heightmap,
        biome_map: &[TerrainType],
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<TerrainFeature>> {
        let mut features = Vec::new();
        
        // Create a map of biome types to feature types
        let mut biome_features: HashMap<TerrainType, Vec<String>> = HashMap::new();
        for biome in &params.biomes {
            biome_features.insert(biome.terrain_type, biome.features.clone());
        }
        
        // Calculate number of features based on terrain size and density
        let feature_count = (params.width * params.height) as f32 * params.feature_density;
        let feature_count = feature_count as usize;
        
        // Use AI for feature placement if available
        if let Some(inference_engine) = &self.inference_engine {
            // Prepare input for AI
            let mut inputs = HashMap::new();
            
            // TODO: Convert heightmap, biome map, etc. to tensors
            
            // Run inference
            let options = InferenceOptions {
                use_cache: true,
                ..Default::default()
            };
            
            let result = inference_engine
                .run_inference(&"terrain-feature-placement".to_string(), inputs, Some(options))
                .await;
            
            // Process AI output if successful
            if let Ok(output) = result {
                // TODO: Parse AI output to get feature placements
                // For now, fall back to random placement
            }
        }
        
        // Fall back to random feature placement
        for _ in 0..feature_count {
            // Pick a random location
            let x = rng.gen_range(0..params.width);
            let y = rng.gen_range(0..params.height);
            let index = y * params.width + x;
            
            // Get the biome at this location
            let biome_type = biome_map[index];
            
            // Skip if underwater
            if biome_type == TerrainType::Ocean {
                continue;
            }
            
            // Get possible features for this biome
            if let Some(possible_features) = biome_features.get(&biome_type) {
                if possible_features.is_empty() {
                    continue;
                }
                
                // Pick a random feature type
                let feature_type = possible_features[rng.gen_range(0..possible_features.len())].clone();
                
                // Create the feature
                let feature = TerrainFeature {
                    feature_type,
                    position: [x as f32, y as f32],
                    rotation: rng.gen_range(0.0..std::f32::consts::PI * 2.0),
                    scale: rng.gen_range(0.8..1.2),
                    parameters: HashMap::new(),
                };
                
                features.push(feature);
            }
        }
        
        Ok(features)
    }
}
