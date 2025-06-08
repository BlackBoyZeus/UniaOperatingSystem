//! Machine learning-based player modeling.
//!
//! This module provides advanced player modeling capabilities using
//! machine learning techniques to understand and predict player behavior.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::error::{AIError, Result};
use crate::game_ai::player::{PlayerId, PlayerAction, PlayerProfile};

/// Player feature vector.
pub type PlayerFeatureVector = Array1<f32>;

/// Player model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerModelType {
    /// Clustering model for player segmentation
    Clustering,
    /// Classification model for play style prediction
    Classification,
    /// Regression model for engagement prediction
    Regression,
    /// Recommendation model for content suggestions
    Recommendation,
    /// Sequence model for behavior prediction
    Sequence,
}

/// Player model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerModelParams {
    /// Model type
    pub model_type: PlayerModelType,
    
    /// Feature names
    pub feature_names: Vec<String>,
    
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Player clustering model.
pub struct PlayerClusteringModel {
    /// Model parameters
    params: PlayerModelParams,
    
    /// Cluster centroids
    centroids: Array2<f32>,
    
    /// Cluster labels
    labels: Vec<String>,
}

impl PlayerClusteringModel {
    /// Create a new player clustering model.
    pub fn new(params: PlayerModelParams, centroids: Array2<f32>, labels: Vec<String>) -> Self {
        Self {
            params,
            centroids,
            labels,
        }
    }
    
    /// Predict the cluster for a player.
    pub fn predict(&self, features: &PlayerFeatureVector) -> Result<(String, f32)> {
        // Find the nearest centroid
        let mut min_distance = f32::MAX;
        let mut nearest_cluster = 0;
        
        for (i, centroid) in self.centroids.axis_iter(Axis(0)).enumerate() {
            // Calculate Euclidean distance
            let distance: f32 = centroid
                .iter()
                .zip(features.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            
            if distance < min_distance {
                min_distance = distance;
                nearest_cluster = i;
            }
        }
        
        // Return the cluster label and confidence (inverse of distance)
        let confidence = 1.0 / (1.0 + min_distance);
        Ok((self.labels[nearest_cluster].clone(), confidence))
    }
}

/// Player classification model.
pub struct PlayerClassificationModel {
    /// Model parameters
    params: PlayerModelParams,
    
    /// Class weights
    weights: Array2<f32>,
    
    /// Class biases
    biases: Array1<f32>,
    
    /// Class labels
    labels: Vec<String>,
}

impl PlayerClassificationModel {
    /// Create a new player classification model.
    pub fn new(
        params: PlayerModelParams,
        weights: Array2<f32>,
        biases: Array1<f32>,
        labels: Vec<String>,
    ) -> Self {
        Self {
            params,
            weights,
            biases,
            labels,
        }
    }
    
    /// Predict the class for a player.
    pub fn predict(&self, features: &PlayerFeatureVector) -> Result<HashMap<String, f32>> {
        // Calculate logits
        let logits = self.weights.dot(features) + &self.biases;
        
        // Apply softmax
        let max_logit = logits.fold(f32::MIN, |a, &b| a.max(b));
        let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let probabilities = exp_logits / sum_exp;
        
        // Create result map
        let mut result = HashMap::new();
        for (i, label) in self.labels.iter().enumerate() {
            result.insert(label.clone(), probabilities[i]);
        }
        
        Ok(result)
    }
}

/// Player regression model.
pub struct PlayerRegressionModel {
    /// Model parameters
    params: PlayerModelParams,
    
    /// Weights
    weights: Array1<f32>,
    
    /// Bias
    bias: f32,
    
    /// Target name
    target_name: String,
}

impl PlayerRegressionModel {
    /// Create a new player regression model.
    pub fn new(
        params: PlayerModelParams,
        weights: Array1<f32>,
        bias: f32,
        target_name: String,
    ) -> Self {
        Self {
            params,
            weights,
            bias,
            target_name,
        }
    }
    
    /// Predict a value for a player.
    pub fn predict(&self, features: &PlayerFeatureVector) -> Result<f32> {
        // Calculate dot product
        let prediction = features.dot(&self.weights) + self.bias;
        
        Ok(prediction)
    }
}

/// Player recommendation model.
pub struct PlayerRecommendationModel {
    /// Model parameters
    params: PlayerModelParams,
    
    /// Item embeddings
    item_embeddings: HashMap<String, Array1<f32>>,
    
    /// Item metadata
    item_metadata: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl PlayerRecommendationModel {
    /// Create a new player recommendation model.
    pub fn new(
        params: PlayerModelParams,
        item_embeddings: HashMap<String, Array1<f32>>,
        item_metadata: HashMap<String, HashMap<String, serde_json::Value>>,
    ) -> Self {
        Self {
            params,
            item_embeddings,
            item_metadata,
        }
    }
    
    /// Recommend items for a player.
    pub fn recommend(
        &self,
        player_embedding: &PlayerFeatureVector,
        count: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Calculate similarity scores
        let mut scores = Vec::new();
        
        for (item_id, item_embedding) in &self.item_embeddings {
            // Calculate cosine similarity
            let dot_product = player_embedding.dot(item_embedding);
            let player_norm = player_embedding.dot(player_embedding).sqrt();
            let item_norm = item_embedding.dot(item_embedding).sqrt();
            
            let similarity = if player_norm > 0.0 && item_norm > 0.0 {
                dot_product / (player_norm * item_norm)
            } else {
                0.0
            };
            
            scores.push((item_id.clone(), similarity));
        }
        
        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top N
        Ok(scores.into_iter().take(count).collect())
    }
}

/// Player sequence model.
pub struct PlayerSequenceModel {
    /// Model parameters
    params: PlayerModelParams,
    
    /// Transition matrix
    transitions: HashMap<String, HashMap<String, f32>>,
    
    /// Action types
    action_types: Vec<String>,
}

impl PlayerSequenceModel {
    /// Create a new player sequence model.
    pub fn new(
        params: PlayerModelParams,
        transitions: HashMap<String, HashMap<String, f32>>,
        action_types: Vec<String>,
    ) -> Self {
        Self {
            params,
            transitions,
            action_types,
        }
    }
    
    /// Predict the next action for a player.
    pub fn predict_next_action(
        &self,
        current_action: &str,
    ) -> Result<HashMap<String, f32>> {
        // Get transition probabilities for the current action
        if let Some(next_actions) = self.transitions.get(current_action) {
            Ok(next_actions.clone())
        } else {
            // If current action is unknown, return uniform distribution
            let uniform_prob = 1.0 / self.action_types.len() as f32;
            let mut result = HashMap::new();
            for action_type in &self.action_types {
                result.insert(action_type.clone(), uniform_prob);
            }
            Ok(result)
        }
    }
}

/// Player model manager.
pub struct PlayerModelManager {
    /// Clustering models
    clustering_models: RwLock<HashMap<String, Arc<PlayerClusteringModel>>>,
    
    /// Classification models
    classification_models: RwLock<HashMap<String, Arc<PlayerClassificationModel>>>,
    
    /// Regression models
    regression_models: RwLock<HashMap<String, Arc<PlayerRegressionModel>>>,
    
    /// Recommendation models
    recommendation_models: RwLock<HashMap<String, Arc<PlayerRecommendationModel>>>,
    
    /// Sequence models
    sequence_models: RwLock<HashMap<String, Arc<PlayerSequenceModel>>>,
    
    /// Player feature cache
    feature_cache: RwLock<HashMap<PlayerId, (PlayerFeatureVector, Instant)>>,
    
    /// Cache TTL
    cache_ttl: Duration,
}

impl PlayerModelManager {
    /// Create a new player model manager.
    pub fn new() -> Self {
        Self {
            clustering_models: RwLock::new(HashMap::new()),
            classification_models: RwLock::new(HashMap::new()),
            regression_models: RwLock::new(HashMap::new()),
            recommendation_models: RwLock::new(HashMap::new()),
            sequence_models: RwLock::new(HashMap::new()),
            feature_cache: RwLock::new(HashMap::new()),
            cache_ttl: Duration::from_secs(60),
        }
    }
    
    /// Register a clustering model.
    pub fn register_clustering_model(
        &self,
        name: &str,
        model: Arc<PlayerClusteringModel>,
    ) {
        self.clustering_models.write().insert(name.to_string(), model);
    }
    
    /// Register a classification model.
    pub fn register_classification_model(
        &self,
        name: &str,
        model: Arc<PlayerClassificationModel>,
    ) {
        self.classification_models.write().insert(name.to_string(), model);
    }
    
    /// Register a regression model.
    pub fn register_regression_model(
        &self,
        name: &str,
        model: Arc<PlayerRegressionModel>,
    ) {
        self.regression_models.write().insert(name.to_string(), model);
    }
    
    /// Register a recommendation model.
    pub fn register_recommendation_model(
        &self,
        name: &str,
        model: Arc<PlayerRecommendationModel>,
    ) {
        self.recommendation_models.write().insert(name.to_string(), model);
    }
    
    /// Register a sequence model.
    pub fn register_sequence_model(
        &self,
        name: &str,
        model: Arc<PlayerSequenceModel>,
    ) {
        self.sequence_models.write().insert(name.to_string(), model);
    }
    
    /// Extract features from a player profile.
    pub fn extract_features(&self, profile: &PlayerProfile) -> Result<PlayerFeatureVector> {
        // This is a simplified implementation for demonstration purposes
        // In a real implementation, this would extract meaningful features from the profile
        
        let mut features = Vec::new();
        
        // Add play style features
        for style in &["explorer", "achiever", "socializer", "killer"] {
            features.push(*profile.play_style.get(*style).unwrap_or(&0.0));
        }
        
        // Add skill features
        for skill in &["combat", "exploration", "puzzle", "social"] {
            features.push(*profile.skills.get(*skill).unwrap_or(&0.0));
        }
        
        // Add preference features
        for pref in &["action", "adventure", "rpg", "strategy", "puzzle"] {
            features.push(*profile.preferences.get(*pref).unwrap_or(&0.0));
        }
        
        // Add session stats
        features.push(profile.session_stats.avg_session_length / 3600.0); // Normalize to hours
        features.push(profile.session_stats.session_count as f32 / 100.0); // Normalize to 0-1 range
        
        // Convert to ndarray
        let feature_vector = Array1::from(features);
        
        Ok(feature_vector)
    }
    
    /// Get or compute features for a player.
    pub fn get_features(&self, profile: &PlayerProfile) -> Result<PlayerFeatureVector> {
        // Check cache first
        let mut cache = self.feature_cache.write();
        
        if let Some((features, timestamp)) = cache.get(&profile.id) {
            // Check if cache is still valid
            if timestamp.elapsed() < self.cache_ttl {
                return Ok(features.clone());
            }
        }
        
        // Extract features
        let features = self.extract_features(profile)?;
        
        // Update cache
        cache.insert(profile.id.clone(), (features.clone(), Instant::now()));
        
        Ok(features)
    }
    
    /// Predict player cluster.
    pub fn predict_cluster(
        &self,
        profile: &PlayerProfile,
        model_name: &str,
    ) -> Result<(String, f32)> {
        // Get features
        let features = self.get_features(profile)?;
        
        // Get model
        let models = self.clustering_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            AIError::ModelNotFound(format!("Clustering model not found: {}", model_name))
        })?;
        
        // Make prediction
        model.predict(&features)
    }
    
    /// Predict player class.
    pub fn predict_class(
        &self,
        profile: &PlayerProfile,
        model_name: &str,
    ) -> Result<HashMap<String, f32>> {
        // Get features
        let features = self.get_features(profile)?;
        
        // Get model
        let models = self.classification_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            AIError::ModelNotFound(format!("Classification model not found: {}", model_name))
        })?;
        
        // Make prediction
        model.predict(&features)
    }
    
    /// Predict player value.
    pub fn predict_value(
        &self,
        profile: &PlayerProfile,
        model_name: &str,
    ) -> Result<f32> {
        // Get features
        let features = self.get_features(profile)?;
        
        // Get model
        let models = self.regression_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            AIError::ModelNotFound(format!("Regression model not found: {}", model_name))
        })?;
        
        // Make prediction
        model.predict(&features)
    }
    
    /// Recommend items for a player.
    pub fn recommend_items(
        &self,
        profile: &PlayerProfile,
        model_name: &str,
        count: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Get features
        let features = self.get_features(profile)?;
        
        // Get model
        let models = self.recommendation_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            AIError::ModelNotFound(format!("Recommendation model not found: {}", model_name))
        })?;
        
        // Make recommendations
        model.recommend(&features, count)
    }
    
    /// Predict next action for a player.
    pub fn predict_next_action(
        &self,
        current_action: &str,
        model_name: &str,
    ) -> Result<HashMap<String, f32>> {
        // Get model
        let models = self.sequence_models.read();
        let model = models.get(model_name).ok_or_else(|| {
            AIError::ModelNotFound(format!("Sequence model not found: {}", model_name))
        })?;
        
        // Make prediction
        model.predict_next_action(current_action)
    }
    
    /// Update models with new player data.
    pub fn update_models(&self, profiles: &[PlayerProfile], actions: &[PlayerAction]) -> Result<()> {
        // This is a placeholder for model updating logic
        // In a real implementation, this would update the models with new data
        
        // For now, just log that we would update the models
        tracing::info!(
            "Would update models with {} profiles and {} actions",
            profiles.len(),
            actions.len()
        );
        
        Ok(())
    }
    
    /// Create default models for testing.
    pub fn create_default_models(&self) -> Result<()> {
        // Create a simple clustering model
        let clustering_params = PlayerModelParams {
            model_type: PlayerModelType::Clustering,
            feature_names: vec![
                "explorer".to_string(),
                "achiever".to_string(),
                "socializer".to_string(),
                "killer".to_string(),
            ],
            parameters: HashMap::new(),
        };
        
        // Define 4 clusters for player types
        let centroids = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.8, 0.3, 0.2, 0.1, // Explorer
                0.3, 0.8, 0.2, 0.1, // Achiever
                0.2, 0.3, 0.8, 0.1, // Socializer
                0.1, 0.2, 0.1, 0.8, // Killer
            ],
        ).map_err(|e| AIError::InternalError(format!("Failed to create centroids: {}", e)))?;
        
        let labels = vec![
            "Explorer".to_string(),
            "Achiever".to_string(),
            "Socializer".to_string(),
            "Killer".to_string(),
        ];
        
        let clustering_model = Arc::new(PlayerClusteringModel::new(
            clustering_params,
            centroids,
            labels,
        ));
        
        self.register_clustering_model("player_types", clustering_model);
        
        // Create a simple classification model
        let classification_params = PlayerModelParams {
            model_type: PlayerModelType::Classification,
            feature_names: vec![
                "explorer".to_string(),
                "achiever".to_string(),
                "socializer".to_string(),
                "killer".to_string(),
                "combat".to_string(),
                "exploration".to_string(),
                "puzzle".to_string(),
                "social".to_string(),
            ],
            parameters: HashMap::new(),
        };
        
        // Simple weights for content preference prediction
        let weights = Array2::from_shape_vec(
            (5, 8),
            vec![
                0.5, 0.2, 0.1, 0.3, 0.8, 0.2, 0.1, 0.0, // Action
                0.7, 0.3, 0.1, 0.1, 0.3, 0.8, 0.2, 0.1, // Adventure
                0.3, 0.7, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, // RPG
                0.1, 0.5, 0.1, 0.3, 0.2, 0.1, 0.8, 0.1, // Strategy
                0.2, 0.3, 0.1, 0.1, 0.1, 0.2, 0.9, 0.1, // Puzzle
            ],
        ).map_err(|e| AIError::InternalError(format!("Failed to create weights: {}", e)))?;
        
        let biases = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]);
        
        let labels = vec![
            "Action".to_string(),
            "Adventure".to_string(),
            "RPG".to_string(),
            "Strategy".to_string(),
            "Puzzle".to_string(),
        ];
        
        let classification_model = Arc::new(PlayerClassificationModel::new(
            classification_params,
            weights,
            biases,
            labels,
        ));
        
        self.register_classification_model("content_preference", classification_model);
        
        // Create a simple regression model
        let regression_params = PlayerModelParams {
            model_type: PlayerModelType::Regression,
            feature_names: vec![
                "explorer".to_string(),
                "achiever".to_string(),
                "socializer".to_string(),
                "killer".to_string(),
                "avg_session_length".to_string(),
                "session_count".to_string(),
            ],
            parameters: HashMap::new(),
        };
        
        // Simple weights for engagement prediction
        let weights = Array1::from_vec(vec![0.2, 0.3, 0.1, 0.1, 0.4, 0.5]);
        let bias = 0.1;
        
        let regression_model = Arc::new(PlayerRegressionModel::new(
            regression_params,
            weights,
            bias,
            "engagement_score".to_string(),
        ));
        
        self.register_regression_model("engagement", regression_model);
        
        // Create a simple recommendation model
        let recommendation_params = PlayerModelParams {
            model_type: PlayerModelType::Recommendation,
            feature_names: vec![
                "explorer".to_string(),
                "achiever".to_string(),
                "socializer".to_string(),
                "killer".to_string(),
            ],
            parameters: HashMap::new(),
        };
        
        // Simple item embeddings
        let mut item_embeddings = HashMap::new();
        item_embeddings.insert("quest_1".to_string(), Array1::from_vec(vec![0.8, 0.3, 0.1, 0.1]));
        item_embeddings.insert("quest_2".to_string(), Array1::from_vec(vec![0.2, 0.9, 0.1, 0.1]));
        item_embeddings.insert("quest_3".to_string(), Array1::from_vec(vec![0.1, 0.1, 0.9, 0.1]));
        item_embeddings.insert("quest_4".to_string(), Array1::from_vec(vec![0.1, 0.1, 0.1, 0.9]));
        
        // Simple item metadata
        let mut item_metadata = HashMap::new();
        
        let mut quest_1_metadata = HashMap::new();
        quest_1_metadata.insert("name".to_string(), serde_json::json!("The Explorer's Path"));
        quest_1_metadata.insert("type".to_string(), serde_json::json!("exploration"));
        item_metadata.insert("quest_1".to_string(), quest_1_metadata);
        
        let mut quest_2_metadata = HashMap::new();
        quest_2_metadata.insert("name".to_string(), serde_json::json!("The Champion's Challenge"));
        quest_2_metadata.insert("type".to_string(), serde_json::json!("combat"));
        item_metadata.insert("quest_2".to_string(), quest_2_metadata);
        
        let mut quest_3_metadata = HashMap::new();
        quest_3_metadata.insert("name".to_string(), serde_json::json!("The Social Network"));
        quest_3_metadata.insert("type".to_string(), serde_json::json!("social"));
        item_metadata.insert("quest_3".to_string(), quest_3_metadata);
        
        let mut quest_4_metadata = HashMap::new();
        quest_4_metadata.insert("name".to_string(), serde_json::json!("The Arena"));
        quest_4_metadata.insert("type".to_string(), serde_json::json!("pvp"));
        item_metadata.insert("quest_4".to_string(), quest_4_metadata);
        
        let recommendation_model = Arc::new(PlayerRecommendationModel::new(
            recommendation_params,
            item_embeddings,
            item_metadata,
        ));
        
        self.register_recommendation_model("quest_recommendations", recommendation_model);
        
        // Create a simple sequence model
        let sequence_params = PlayerModelParams {
            model_type: PlayerModelType::Sequence,
            feature_names: Vec::new(),
            parameters: HashMap::new(),
        };
        
        // Simple transition matrix
        let mut transitions = HashMap::new();
        
        let mut explore_transitions = HashMap::new();
        explore_transitions.insert("explore".to_string(), 0.6);
        explore_transitions.insert("combat".to_string(), 0.2);
        explore_transitions.insert("interact".to_string(), 0.2);
        transitions.insert("explore".to_string(), explore_transitions);
        
        let mut combat_transitions = HashMap::new();
        combat_transitions.insert("explore".to_string(), 0.3);
        combat_transitions.insert("combat".to_string(), 0.5);
        combat_transitions.insert("interact".to_string(), 0.2);
        transitions.insert("combat".to_string(), combat_transitions);
        
        let mut interact_transitions = HashMap::new();
        interact_transitions.insert("explore".to_string(), 0.4);
        interact_transitions.insert("combat".to_string(), 0.2);
        interact_transitions.insert("interact".to_string(), 0.4);
        transitions.insert("interact".to_string(), interact_transitions);
        
        let action_types = vec![
            "explore".to_string(),
            "combat".to_string(),
            "interact".to_string(),
        ];
        
        let sequence_model = Arc::new(PlayerSequenceModel::new(
            sequence_params,
            transitions,
            action_types,
        ));
        
        self.register_sequence_model("action_sequence", sequence_model);
        
        Ok(())
    }
}
