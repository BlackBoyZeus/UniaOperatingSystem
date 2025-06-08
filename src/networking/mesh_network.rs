//! Mesh networking implementation for UNIA.
//!
//! This module provides a WebRTC-based mesh networking system for
//! multiplayer gaming with AI integration.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::channel::mpsc::{self, UnboundedReceiver, UnboundedSender};
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time;
use uuid::Uuid;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::{APIBuilder, API};
use webrtc::data_channel::data_channel_init::RTCDataChannelInit;
use webrtc::data_channel::RTCDataChannel;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;

use crate::error::{AIError, Result};
use crate::game_ai::GameAIManager;

/// Mesh network node ID.
pub type NodeId = String;

/// Mesh network message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Message ID
    pub id: String,
    
    /// Sender node ID
    pub sender: NodeId,
    
    /// Recipient node ID (None for broadcast)
    pub recipient: Option<NodeId>,
    
    /// Message type
    pub message_type: String,
    
    /// Message payload
    pub payload: Vec<u8>,
    
    /// Message timestamp
    pub timestamp: u64,
    
    /// Time-to-live (for broadcast messages)
    pub ttl: u8,
}

impl NetworkMessage {
    /// Create a new network message.
    pub fn new(
        sender: NodeId,
        recipient: Option<NodeId>,
        message_type: &str,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            sender,
            recipient,
            message_type: message_type.to_string(),
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ttl: 5,
        }
    }
}

/// Network event.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Node connected
    NodeConnected(NodeId),
    
    /// Node disconnected
    NodeDisconnected(NodeId),
    
    /// Message received
    MessageReceived(NetworkMessage),
    
    /// Connection state changed
    ConnectionStateChanged(RTCPeerConnectionState),
    
    /// Error occurred
    Error(String),
}

/// Network event handler.
#[async_trait]
pub trait NetworkEventHandler: Send + Sync {
    /// Handle a network event.
    async fn handle_event(&self, event: NetworkEvent);
}

/// Mesh network configuration.
#[derive(Debug, Clone)]
pub struct MeshNetworkConfig {
    /// Node ID
    pub node_id: NodeId,
    
    /// STUN servers
    pub stun_servers: Vec<String>,
    
    /// TURN servers
    pub turn_servers: Vec<(String, String, String)>, // (url, username, credential)
    
    /// Maximum number of peers
    pub max_peers: usize,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
}

impl Default for MeshNetworkConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4().to_string(),
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            max_peers: 32,
            connection_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(5),
        }
    }
}

/// Mesh network for multiplayer gaming.
pub struct MeshNetwork {
    /// Network configuration
    config: MeshNetworkConfig,
    
    /// WebRTC API
    api: API,
    
    /// Peer connections
    peer_connections: RwLock<HashMap<NodeId, Arc<RTCPeerConnection>>>,
    
    /// Data channels
    data_channels: RwLock<HashMap<NodeId, Arc<RTCDataChannel>>>,
    
    /// Connected peers
    connected_peers: RwLock<HashSet<NodeId>>,
    
    /// Event handlers
    event_handlers: RwLock<Vec<Box<dyn NetworkEventHandler>>>,
    
    /// Event sender
    event_sender: UnboundedSender<NetworkEvent>,
    
    /// Event receiver
    event_receiver: Mutex<Option<UnboundedReceiver<NetworkEvent>>>,
    
    /// Game AI manager
    ai_manager: Option<Arc<GameAIManager>>,
}

impl MeshNetwork {
    /// Create a new mesh network.
    ///
    /// # Arguments
    ///
    /// * `config` - Network configuration
    /// * `ai_manager` - Game AI manager (optional)
    ///
    /// # Returns
    ///
    /// A Result containing the initialized mesh network or an error
    pub async fn new(
        config: MeshNetworkConfig,
        ai_manager: Option<Arc<GameAIManager>>,
    ) -> Result<Arc<Self>> {
        // Create WebRTC API
        let mut media_engine = MediaEngine::default();
        let mut registry = register_default_interceptors();
        
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .build();
        
        // Create event channel
        let (event_sender, event_receiver) = mpsc::unbounded();
        
        let network = Arc::new(Self {
            config,
            api,
            peer_connections: RwLock::new(HashMap::new()),
            data_channels: RwLock::new(HashMap::new()),
            connected_peers: RwLock::new(HashSet::new()),
            event_handlers: RwLock::new(Vec::new()),
            event_sender,
            event_receiver: Mutex::new(Some(event_receiver)),
            ai_manager,
        });
        
        // Start event processing
        Self::start_event_processing(Arc::clone(&network));
        
        // Start heartbeat
        Self::start_heartbeat(Arc::clone(&network));
        
        Ok(network)
    }
    
    /// Start event processing.
    fn start_event_processing(network: Arc<Self>) {
        tokio::spawn(async move {
            let mut receiver = network.event_receiver.lock().await.take().unwrap();
            
            while let Some(event) = receiver.next().await {
                // Notify event handlers
                for handler in network.event_handlers.read().iter() {
                    handler.handle_event(event.clone()).await;
                }
                
                // Process the event
                match &event {
                    NetworkEvent::NodeConnected(node_id) => {
                        network.connected_peers.write().insert(node_id.clone());
                        tracing::info!("Node connected: {}", node_id);
                    }
                    NetworkEvent::NodeDisconnected(node_id) => {
                        network.connected_peers.write().remove(node_id);
                        network.peer_connections.write().remove(node_id);
                        network.data_channels.write().remove(node_id);
                        tracing::info!("Node disconnected: {}", node_id);
                    }
                    NetworkEvent::MessageReceived(message) => {
                        // Handle message
                        if let Err(e) = network.handle_message(message).await {
                            tracing::error!("Failed to handle message: {}", e);
                        }
                    }
                    NetworkEvent::ConnectionStateChanged(state) => {
                        tracing::debug!("Connection state changed: {:?}", state);
                    }
                    NetworkEvent::Error(error) => {
                        tracing::error!("Network error: {}", error);
                    }
                }
            }
        });
    }
    
    /// Start heartbeat.
    fn start_heartbeat(network: Arc<Self>) {
        tokio::spawn(async move {
            let interval = network.config.heartbeat_interval;
            let mut ticker = time::interval(interval);
            
            loop {
                ticker.tick().await;
                
                // Send heartbeat to all connected peers
                let connected_peers: Vec<NodeId> = network.connected_peers.read().iter().cloned().collect();
                
                for peer_id in connected_peers {
                    let heartbeat = NetworkMessage::new(
                        network.config.node_id.clone(),
                        Some(peer_id.clone()),
                        "heartbeat",
                        Vec::new(),
                    );
                    
                    if let Err(e) = network.send_message(&heartbeat).await {
                        tracing::warn!("Failed to send heartbeat to {}: {}", peer_id, e);
                    }
                }
            }
        });
    }
    
    /// Register an event handler.
    ///
    /// # Arguments
    ///
    /// * `handler` - Event handler
    pub fn register_event_handler(&self, handler: Box<dyn NetworkEventHandler>) {
        self.event_handlers.write().push(handler);
    }
    
    /// Connect to a peer.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Peer node ID
    /// * `signaling_data` - Signaling data for connection establishment
    ///
    /// # Returns
    ///
    /// A Result containing the connection offer or an error
    pub async fn connect_to_peer(
        self: &Arc<Self>,
        peer_id: &str,
        signaling_data: Option<String>,
    ) -> Result<String> {
        // Check if we're already connected to this peer
        if self.connected_peers.read().contains(peer_id) {
            return Err(AIError::NetworkError(format!(
                "Already connected to peer: {}",
                peer_id
            )));
        }
        
        // Check if we've reached the maximum number of peers
        if self.connected_peers.read().len() >= self.config.max_peers {
            return Err(AIError::NetworkError(
                "Maximum number of peers reached".to_string(),
            ));
        }
        
        // Create peer connection configuration
        let mut ice_servers = Vec::new();
        
        // Add STUN servers
        for stun_server in &self.config.stun_servers {
            ice_servers.push(RTCIceServer {
                urls: vec![stun_server.clone()],
                ..Default::default()
            });
        }
        
        // Add TURN servers
        for (url, username, credential) in &self.config.turn_servers {
            ice_servers.push(RTCIceServer {
                urls: vec![url.clone()],
                username: username.clone(),
                credential: credential.clone(),
                ..Default::default()
            });
        }
        
        let config = RTCConfiguration {
            ice_servers,
            ..Default::default()
        };
        
        // Create peer connection
        let peer_connection = self.api.new_peer_connection(config).await
            .map_err(|e| AIError::NetworkError(format!("Failed to create peer connection: {}", e)))?;
        
        // Set up data channel
        let data_channel_init = RTCDataChannelInit {
            ordered: Some(true),
            ..Default::default()
        };
        
        let data_channel = peer_connection.create_data_channel("data", Some(data_channel_init)).await
            .map_err(|e| AIError::NetworkError(format!("Failed to create data channel: {}", e)))?;
        
        let peer_id_clone = peer_id.to_string();
        let network = Arc::clone(self);
        
        // Set up data channel callbacks
        data_channel.on_open(Box::new(move || {
            let peer_id = peer_id_clone.clone();
            let network = Arc::clone(&network);
            
            Box::pin(async move {
                // Notify that the node is connected
                if let Err(e) = network.event_sender.clone().send(NetworkEvent::NodeConnected(peer_id)).await {
                    tracing::error!("Failed to send NodeConnected event: {}", e);
                }
            })
        }));
        
        let peer_id_clone = peer_id.to_string();
        let network = Arc::clone(self);
        
        data_channel.on_message(Box::new(move |msg| {
            let peer_id = peer_id_clone.clone();
            let network = Arc::clone(&network);
            let msg_data = msg.data.to_vec();
            
            Box::pin(async move {
                // Parse the message
                match serde_json::from_slice::<NetworkMessage>(&msg_data) {
                    Ok(message) => {
                        // Notify that a message was received
                        if let Err(e) = network.event_sender.clone().send(NetworkEvent::MessageReceived(message)).await {
                            tracing::error!("Failed to send MessageReceived event: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse message from {}: {}", peer_id, e);
                    }
                }
            })
        }));
        
        let peer_id_clone = peer_id.to_string();
        let network = Arc::clone(self);
        
        data_channel.on_close(Box::new(move || {
            let peer_id = peer_id_clone.clone();
            let network = Arc::clone(&network);
            
            Box::pin(async move {
                // Notify that the node is disconnected
                if let Err(e) = network.event_sender.clone().send(NetworkEvent::NodeDisconnected(peer_id)).await {
                    tracing::error!("Failed to send NodeDisconnected event: {}", e);
                }
            })
        }));
        
        // Set up peer connection callbacks
        let peer_id_clone = peer_id.to_string();
        let network = Arc::clone(self);
        
        peer_connection.on_peer_connection_state_change(Box::new(move |state| {
            let peer_id = peer_id_clone.clone();
            let network = Arc::clone(&network);
            
            Box::pin(async move {
                // Notify that the connection state changed
                if let Err(e) = network.event_sender.clone().send(NetworkEvent::ConnectionStateChanged(state)).await {
                    tracing::error!("Failed to send ConnectionStateChanged event: {}", e);
                }
                
                // Handle disconnection
                if state == RTCPeerConnectionState::Disconnected || state == RTCPeerConnectionState::Failed || state == RTCPeerConnectionState::Closed {
                    if let Err(e) = network.event_sender.clone().send(NetworkEvent::NodeDisconnected(peer_id)).await {
                        tracing::error!("Failed to send NodeDisconnected event: {}", e);
                    }
                }
            })
        }));
        
        // Store peer connection and data channel
        self.peer_connections.write().insert(peer_id.to_string(), Arc::new(peer_connection.clone()));
        self.data_channels.write().insert(peer_id.to_string(), Arc::new(data_channel));
        
        // Create offer or process signaling data
        let signaling_result = if let Some(signaling_data) = signaling_data {
            // Process signaling data (answer)
            let answer: RTCSessionDescription = serde_json::from_str(&signaling_data)
                .map_err(|e| AIError::NetworkError(format!("Failed to parse signaling data: {}", e)))?;
            
            peer_connection.set_remote_description(answer).await
                .map_err(|e| AIError::NetworkError(format!("Failed to set remote description: {}", e)))?;
            
            // Return empty string since we've processed the answer
            "".to_string()
        } else {
            // Create offer
            let offer = peer_connection.create_offer(None).await
                .map_err(|e| AIError::NetworkError(format!("Failed to create offer: {}", e)))?;
            
            peer_connection.set_local_description(offer.clone()).await
                .map_err(|e| AIError::NetworkError(format!("Failed to set local description: {}", e)))?;
            
            // Return the offer as signaling data
            serde_json::to_string(&offer)
                .map_err(|e| AIError::NetworkError(format!("Failed to serialize offer: {}", e)))?
        };
        
        Ok(signaling_result)
    }
    
    /// Process signaling data from a peer.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Peer node ID
    /// * `signaling_data` - Signaling data
    ///
    /// # Returns
    ///
    /// A Result containing the response signaling data or an error
    pub async fn process_signaling(
        &self,
        peer_id: &str,
        signaling_data: &str,
    ) -> Result<String> {
        // Get peer connection
        let peer_connections = self.peer_connections.read();
        let peer_connection = peer_connections.get(peer_id).cloned().ok_or_else(|| {
            AIError::NetworkError(format!("No peer connection for peer: {}", peer_id))
        })?;
        
        // Parse signaling data
        let session_description: RTCSessionDescription = serde_json::from_str(signaling_data)
            .map_err(|e| AIError::NetworkError(format!("Failed to parse signaling data: {}", e)))?;
        
        // Process offer or answer
        if session_description.sdp_type.to_string() == "offer" {
            // Set remote description
            peer_connection.set_remote_description(session_description).await
                .map_err(|e| AIError::NetworkError(format!("Failed to set remote description: {}", e)))?;
            
            // Create answer
            let answer = peer_connection.create_answer(None).await
                .map_err(|e| AIError::NetworkError(format!("Failed to create answer: {}", e)))?;
            
            // Set local description
            peer_connection.set_local_description(answer.clone()).await
                .map_err(|e| AIError::NetworkError(format!("Failed to set local description: {}", e)))?;
            
            // Return the answer as signaling data
            serde_json::to_string(&answer)
                .map_err(|e| AIError::NetworkError(format!("Failed to serialize answer: {}", e)))?
        } else {
            // Set remote description (answer)
            peer_connection.set_remote_description(session_description).await
                .map_err(|e| AIError::NetworkError(format!("Failed to set remote description: {}", e)))?;
            
            // Return empty string since we've processed the answer
            "".to_string()
        };
        
        Ok("".to_string())
    }
    
    /// Send a message to a peer.
    ///
    /// # Arguments
    ///
    /// * `message` - Network message
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub async fn send_message(&self, message: &NetworkMessage) -> Result<()> {
        // Serialize the message
        let message_data = serde_json::to_vec(message)
            .map_err(|e| AIError::NetworkError(format!("Failed to serialize message: {}", e)))?;
        
        // Check if this is a broadcast message
        if message.recipient.is_none() {
            // Send to all connected peers
            let connected_peers: Vec<NodeId> = self.connected_peers.read().iter().cloned().collect();
            
            for peer_id in connected_peers {
                // Skip sending to the sender
                if peer_id == message.sender {
                    continue;
                }
                
                // Create a directed message
                let directed_message = NetworkMessage {
                    recipient: Some(peer_id.clone()),
                    ttl: message.ttl - 1,
                    ..message.clone()
                };
                
                // Send the message
                if let Err(e) = self.send_message(&directed_message).await {
                    tracing::warn!("Failed to send broadcast message to {}: {}", peer_id, e);
                }
            }
            
            return Ok(());
        }
        
        // Get the recipient
        let recipient = message.recipient.as_ref().ok_or_else(|| {
            AIError::NetworkError("Message has no recipient".to_string())
        })?;
        
        // Get the data channel
        let data_channels = self.data_channels.read();
        let data_channel = data_channels.get(recipient).cloned().ok_or_else(|| {
            AIError::NetworkError(format!("No data channel for peer: {}", recipient))
        })?;
        
        // Send the message
        data_channel.send(&message_data).await
            .map_err(|e| AIError::NetworkError(format!("Failed to send message: {}", e)))?;
        
        Ok(())
    }
    
    /// Handle a received message.
    ///
    /// # Arguments
    ///
    /// * `message` - Network message
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    async fn handle_message(&self, message: &NetworkMessage) -> Result<()> {
        // Check if we are the recipient
        if let Some(recipient) = &message.recipient {
            if recipient != &self.config.node_id {
                // Forward the message if TTL > 0
                if message.ttl > 0 {
                    return self.send_message(message).await;
                }
                
                return Ok(());
            }
        }
        
        // Process the message based on its type
        match message.message_type.as_str() {
            "heartbeat" => {
                // Respond to heartbeat
                let response = NetworkMessage::new(
                    self.config.node_id.clone(),
                    Some(message.sender.clone()),
                    "heartbeat_ack",
                    Vec::new(),
                );
                
                self.send_message(&response).await?;
            }
            "heartbeat_ack" => {
                // Nothing to do
            }
            "game_state" => {
                // Process game state update
                // This would typically be handled by the game logic
            }
            "ai_request" => {
                // Process AI request
                if let Some(ai_manager) = &self.ai_manager {
                    // Parse the AI request
                    // This is a simplified implementation for demonstration purposes
                    // In a real implementation, this would properly parse and process the AI request
                }
            }
            _ => {
                // Unknown message type
                tracing::warn!("Received unknown message type: {}", message.message_type);
            }
        }
        
        Ok(())
    }
    
    /// Disconnect from a peer.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Peer node ID
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error
    pub async fn disconnect_from_peer(&self, peer_id: &str) -> Result<()> {
        // Check if we're connected to this peer
        if !self.connected_peers.read().contains(peer_id) {
            return Err(AIError::NetworkError(format!(
                "Not connected to peer: {}",
                peer_id
            )));
        }
        
        // Close the data channel
        if let Some(data_channel) = self.data_channels.write().remove(peer_id) {
            data_channel.close().await
                .map_err(|e| AIError::NetworkError(format!("Failed to close data channel: {}", e)))?;
        }
        
        // Close the peer connection
        if let Some(peer_connection) = self.peer_connections.write().remove(peer_id) {
            peer_connection.close().await
                .map_err(|e| AIError::NetworkError(format!("Failed to close peer connection: {}", e)))?;
        }
        
        // Remove from connected peers
        self.connected_peers.write().remove(peer_id);
        
        Ok(())
    }
    
    /// Get connected peers.
    ///
    /// # Returns
    ///
    /// A vector of connected peer IDs
    pub fn get_connected_peers(&self) -> Vec<NodeId> {
        self.connected_peers.read().iter().cloned().collect()
    }
    
    /// Check if connected to a peer.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Peer node ID
    ///
    /// # Returns
    ///
    /// True if connected to the peer, false otherwise
    pub fn is_connected_to_peer(&self, peer_id: &str) -> bool {
        self.connected_peers.read().contains(peer_id)
    }
    
    /// Get the node ID.
    ///
    /// # Returns
    ///
    /// The node ID
    pub fn get_node_id(&self) -> &str {
        &self.config.node_id
    }
}
