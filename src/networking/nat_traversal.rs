//! NAT traversal for peer-to-peer connections.
//!
//! This module provides NAT traversal capabilities for establishing
//! peer-to-peer connections in multiplayer gaming scenarios.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;
use uuid::Uuid;

use crate::error::{AIError, Result};

/// NAT type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NATType {
    /// Open internet (no NAT)
    Open,
    /// Full cone NAT
    FullCone,
    /// Restricted cone NAT
    RestrictedCone,
    /// Port restricted cone NAT
    PortRestrictedCone,
    /// Symmetric NAT
    Symmetric,
    /// Unknown NAT type
    Unknown,
}

/// STUN message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum STUNMessageType {
    /// Binding request
    BindingRequest,
    /// Binding response
    BindingResponse,
    /// Binding error response
    BindingErrorResponse,
}

/// STUN message.
#[derive(Debug, Clone)]
pub struct STUNMessage {
    /// Message type
    pub message_type: STUNMessageType,
    
    /// Transaction ID
    pub transaction_id: [u8; 12],
    
    /// Mapped address
    pub mapped_address: Option<SocketAddr>,
    
    /// Changed address
    pub changed_address: Option<SocketAddr>,
    
    /// Source address
    pub source_address: Option<SocketAddr>,
    
    /// Changed port
    pub changed_port: Option<u16>,
    
    /// Error code
    pub error_code: Option<u16>,
    
    /// Error message
    pub error_message: Option<String>,
}

/// TURN allocation.
#[derive(Debug, Clone)]
pub struct TURNAllocation {
    /// Allocation ID
    pub id: String,
    
    /// Relayed address
    pub relayed_address: SocketAddr,
    
    /// Server address
    pub server_address: SocketAddr,
    
    /// Username
    pub username: String,
    
    /// Password
    pub password: String,
    
    /// Creation time
    pub created_at: Instant,
    
    /// Expiration time
    pub expires_at: Instant,
}

/// NAT traversal method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraversalMethod {
    /// Direct connection
    Direct,
    /// Hole punching
    HolePunching,
    /// TURN relay
    TURNRelay,
}

/// Connection candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionCandidate {
    /// Candidate ID
    pub id: String,
    
    /// IP address
    pub ip: String,
    
    /// Port
    pub port: u16,
    
    /// Protocol
    pub protocol: String,
    
    /// Priority
    pub priority: u32,
    
    /// Type
    pub candidate_type: String,
    
    /// Related address
    pub related_address: Option<String>,
    
    /// Related port
    pub related_port: Option<u16>,
}

/// NAT traversal configuration.
#[derive(Debug, Clone)]
pub struct NATTraversalConfig {
    /// STUN servers
    pub stun_servers: Vec<String>,
    
    /// TURN servers
    pub turn_servers: Vec<(String, String, String)>, // (url, username, credential)
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Hole punching timeout
    pub hole_punching_timeout: Duration,
    
    /// TURN allocation lifetime
    pub turn_allocation_lifetime: Duration,
    
    /// Preferred traversal methods in order of preference
    pub preferred_methods: Vec<TraversalMethod>,
}

impl Default for NATTraversalConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            connection_timeout: Duration::from_secs(30),
            hole_punching_timeout: Duration::from_secs(10),
            turn_allocation_lifetime: Duration::from_secs(600), // 10 minutes
            preferred_methods: vec![
                TraversalMethod::Direct,
                TraversalMethod::HolePunching,
                TraversalMethod::TURNRelay,
            ],
        }
    }
}

/// NAT traversal event.
#[derive(Debug, Clone)]
pub enum NATTraversalEvent {
    /// NAT type detected
    NATTypeDetected(NATType),
    
    /// External address detected
    ExternalAddressDetected(SocketAddr),
    
    /// Connection established
    ConnectionEstablished {
        /// Peer ID
        peer_id: String,
        
        /// Local address
        local_address: SocketAddr,
        
        /// Remote address
        remote_address: SocketAddr,
        
        /// Traversal method used
        method: TraversalMethod,
    },
    
    /// Connection failed
    ConnectionFailed {
        /// Peer ID
        peer_id: String,
        
        /// Error message
        error: String,
    },
    
    /// TURN allocation created
    TURNAllocationCreated(TURNAllocation),
    
    /// TURN allocation failed
    TURNAllocationFailed {
        /// Server address
        server_address: String,
        
        /// Error message
        error: String,
    },
}

/// NAT traversal event handler.
#[async_trait]
pub trait NATTraversalEventHandler: Send + Sync {
    /// Handle a NAT traversal event.
    async fn handle_event(&self, event: NATTraversalEvent);
}

/// NAT traversal manager.
pub struct NATTraversalManager {
    /// Configuration
    config: NATTraversalConfig,
    
    /// Local NAT type
    nat_type: RwLock<NATType>,
    
    /// External address
    external_address: RwLock<Option<SocketAddr>>,
    
    /// TURN allocations
    turn_allocations: RwLock<HashMap<String, TURNAllocation>>,
    
    /// Event handlers
    event_handlers: RwLock<Vec<Box<dyn NATTraversalEventHandler>>>,
    
    /// Event sender
    event_sender: mpsc::UnboundedSender<NATTraversalEvent>,
    
    /// Event receiver
    event_receiver: RwLock<Option<mpsc::UnboundedReceiver<NATTraversalEvent>>>,
    
    /// UDP socket
    socket: Arc<UdpSocket>,
}

impl NATTraversalManager {
    /// Create a new NAT traversal manager.
    ///
    /// # Arguments
    ///
    /// * `config` - NAT traversal configuration
    ///
    /// # Returns
    ///
    /// A Result containing the initialized NAT traversal manager or an error
    pub async fn new(config: NATTraversalConfig) -> Result<Arc<Self>> {
        // Create UDP socket
        let socket = UdpSocket::bind("0.0.0.0:0").await
            .map_err(|e| AIError::NetworkError(format!("Failed to bind UDP socket: {}", e)))?;
        
        // Create event channel
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        let manager = Arc::new(Self {
            config,
            nat_type: RwLock::new(NATType::Unknown),
            external_address: RwLock::new(None),
            turn_allocations: RwLock::new(HashMap::new()),
            event_handlers: RwLock::new(Vec::new()),
            event_sender,
            event_receiver: RwLock::new(Some(event_receiver)),
            socket: Arc::new(socket),
        });
        
        // Start event processing
        Self::start_event_processing(Arc::clone(&manager));
        
        // Detect NAT type
        manager.detect_nat_type().await?;
        
        Ok(manager)
    }
    
    /// Start event processing.
    fn start_event_processing(manager: Arc<Self>) {
        tokio::spawn(async move {
            let mut receiver = manager.event_receiver.write().take().unwrap();
            
            while let Some(event) = receiver.recv().await {
                // Notify event handlers
                for handler in manager.event_handlers.read().iter() {
                    handler.handle_event(event.clone()).await;
                }
                
                // Process the event
                match &event {
                    NATTraversalEvent::NATTypeDetected(nat_type) => {
                        *manager.nat_type.write() = *nat_type;
                        tracing::info!("NAT type detected: {:?}", nat_type);
                    }
                    NATTraversalEvent::ExternalAddressDetected(address) => {
                        *manager.external_address.write() = Some(*address);
                        tracing::info!("External address detected: {}", address);
                    }
                    NATTraversalEvent::ConnectionEstablished { peer_id, local_address, remote_address, method } => {
                        tracing::info!(
                            "Connection established with {}: {} -> {} using {:?}",
                            peer_id, local_address, remote_address, method
                        );
                    }
                    NATTraversalEvent::ConnectionFailed { peer_id, error } => {
                        tracing::error!("Connection failed with {}: {}", peer_id, error);
                    }
                    NATTraversalEvent::TURNAllocationCreated(allocation) => {
                        manager.turn_allocations.write().insert(allocation.id.clone(), allocation.clone());
                        tracing::info!(
                            "TURN allocation created: {} -> {}",
                            allocation.relayed_address, allocation.server_address
                        );
                    }
                    NATTraversalEvent::TURNAllocationFailed { server_address, error } => {
                        tracing::error!("TURN allocation failed for {}: {}", server_address, error);
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
    pub fn register_event_handler(&self, handler: Box<dyn NATTraversalEventHandler>) {
        self.event_handlers.write().push(handler);
    }
    
    /// Detect NAT type.
    ///
    /// # Returns
    ///
    /// A Result containing the detected NAT type or an error
    pub async fn detect_nat_type(&self) -> Result<NATType> {
        // This is a simplified implementation for demonstration purposes
        // In a real implementation, this would use the STUN protocol to detect the NAT type
        
        // For now, just use the first STUN server
        if let Some(stun_server) = self.config.stun_servers.first() {
            // Parse STUN server address
            let server_addr = parse_stun_server(stun_server)?;
            
            // Send STUN binding request
            let request = create_stun_binding_request();
            
            // Send the request
            self.socket.send_to(&request, server_addr).await
                .map_err(|e| AIError::NetworkError(format!("Failed to send STUN request: {}", e)))?;
            
            // Receive response
            let mut buf = [0u8; 1024];
            let (len, addr) = self.socket.recv_from(&mut buf).await
                .map_err(|e| AIError::NetworkError(format!("Failed to receive STUN response: {}", e)))?;
            
            // Parse response
            let response = parse_stun_response(&buf[..len])?;
            
            // Extract mapped address
            if let Some(mapped_address) = response.mapped_address {
                // Notify about external address
                self.event_sender.send(NATTraversalEvent::ExternalAddressDetected(mapped_address))
                    .map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
                
                // Determine NAT type
                let nat_type = if mapped_address.ip() == get_local_ip()? {
                    NATType::Open
                } else {
                    // This is a simplified detection
                    // In a real implementation, we would perform additional tests
                    NATType::FullCone
                };
                
                // Notify about NAT type
                self.event_sender.send(NATTraversalEvent::NATTypeDetected(nat_type))
                    .map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
                
                return Ok(nat_type);
            }
        }
        
        // Default to unknown
        let nat_type = NATType::Unknown;
        
        // Notify about NAT type
        self.event_sender.send(NATTraversalEvent::NATTypeDetected(nat_type))
            .map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
        
        Ok(nat_type)
    }
    
    /// Get the local NAT type.
    ///
    /// # Returns
    ///
    /// The detected NAT type
    pub fn get_nat_type(&self) -> NATType {
        *self.nat_type.read()
    }
    
    /// Get the external address.
    ///
    /// # Returns
    ///
    /// The detected external address, if available
    pub fn get_external_address(&self) -> Option<SocketAddr> {
        *self.external_address.read()
    }
    
    /// Create connection candidates.
    ///
    /// # Returns
    ///
    /// A Result containing a list of connection candidates or an error
    pub async fn create_connection_candidates(&self) -> Result<Vec<ConnectionCandidate>> {
        let mut candidates = Vec::new();
        
        // Add local candidate
        let local_addr = self.socket.local_addr()
            .map_err(|e| AIError::NetworkError(format!("Failed to get local address: {}", e)))?;
        
        candidates.push(ConnectionCandidate {
            id: Uuid::new_v4().to_string(),
            ip: local_addr.ip().to_string(),
            port: local_addr.port(),
            protocol: "udp".to_string(),
            priority: 1,
            candidate_type: "host".to_string(),
            related_address: None,
            related_port: None,
        });
        
        // Add server-reflexive candidate (from STUN)
        if let Some(external_addr) = *self.external_address.read() {
            candidates.push(ConnectionCandidate {
                id: Uuid::new_v4().to_string(),
                ip: external_addr.ip().to_string(),
                port: external_addr.port(),
                protocol: "udp".to_string(),
                priority: 2,
                candidate_type: "srflx".to_string(),
                related_address: Some(local_addr.ip().to_string()),
                related_port: Some(local_addr.port()),
            });
        }
        
        // Add relay candidate (from TURN)
        for allocation in self.turn_allocations.read().values() {
            candidates.push(ConnectionCandidate {
                id: Uuid::new_v4().to_string(),
                ip: allocation.relayed_address.ip().to_string(),
                port: allocation.relayed_address.port(),
                protocol: "udp".to_string(),
                priority: 3,
                candidate_type: "relay".to_string(),
                related_address: Some(allocation.server_address.ip().to_string()),
                related_port: Some(allocation.server_address.port()),
            });
        }
        
        Ok(candidates)
    }
    
    /// Connect to a peer.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - Peer ID
    /// * `candidates` - Peer's connection candidates
    ///
    /// # Returns
    ///
    /// A Result containing the established connection or an error
    pub async fn connect_to_peer(
        &self,
        peer_id: &str,
        candidates: Vec<ConnectionCandidate>,
    ) -> Result<(SocketAddr, SocketAddr, TraversalMethod)> {
        // Try each traversal method in order of preference
        for method in &self.config.preferred_methods {
            match method {
                TraversalMethod::Direct => {
                    if let Ok((local, remote)) = self.try_direct_connection(peer_id, &candidates).await {
                        // Notify about connection
                        self.event_sender.send(NATTraversalEvent::ConnectionEstablished {
                            peer_id: peer_id.to_string(),
                            local_address: local,
                            remote_address: remote,
                            method: TraversalMethod::Direct,
                        }).map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
                        
                        return Ok((local, remote, TraversalMethod::Direct));
                    }
                }
                TraversalMethod::HolePunching => {
                    if let Ok((local, remote)) = self.try_hole_punching(peer_id, &candidates).await {
                        // Notify about connection
                        self.event_sender.send(NATTraversalEvent::ConnectionEstablished {
                            peer_id: peer_id.to_string(),
                            local_address: local,
                            remote_address: remote,
                            method: TraversalMethod::HolePunching,
                        }).map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
                        
                        return Ok((local, remote, TraversalMethod::HolePunching));
                    }
                }
                TraversalMethod::TURNRelay => {
                    if let Ok((local, remote)) = self.try_turn_relay(peer_id, &candidates).await {
                        // Notify about connection
                        self.event_sender.send(NATTraversalEvent::ConnectionEstablished {
                            peer_id: peer_id.to_string(),
                            local_address: local,
                            remote_address: remote,
                            method: TraversalMethod::TURNRelay,
                        }).map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
                        
                        return Ok((local, remote, TraversalMethod::TURNRelay));
                    }
                }
            }
        }
        
        // All methods failed
        let error = "All connection methods failed".to_string();
        
        // Notify about failure
        self.event_sender.send(NATTraversalEvent::ConnectionFailed {
            peer_id: peer_id.to_string(),
            error: error.clone(),
        }).map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
        
        Err(AIError::NetworkError(error))
    }
    
    /// Try direct connection.
    async fn try_direct_connection(
        &self,
        peer_id: &str,
        candidates: &[ConnectionCandidate],
    ) -> Result<(SocketAddr, SocketAddr)> {
        // Get local address
        let local_addr = self.socket.local_addr()
            .map_err(|e| AIError::NetworkError(format!("Failed to get local address: {}", e)))?;
        
        // Try each candidate
        for candidate in candidates {
            // Skip non-host candidates
            if candidate.candidate_type != "host" {
                continue;
            }
            
            // Parse address
            let addr = format!("{}:{}", candidate.ip, candidate.port).parse::<SocketAddr>()
                .map_err(|e| AIError::NetworkError(format!("Failed to parse address: {}", e)))?;
            
            // Send ping
            let ping = format!("PING:{}", peer_id);
            self.socket.send_to(ping.as_bytes(), addr).await
                .map_err(|e| AIError::NetworkError(format!("Failed to send ping: {}", e)))?;
            
            // Wait for pong
            let mut buf = [0u8; 1024];
            let timeout = time::timeout(Duration::from_secs(2), self.socket.recv_from(&mut buf)).await;
            
            if let Ok(Ok((len, from))) = timeout {
                let message = String::from_utf8_lossy(&buf[..len]);
                if message.starts_with("PONG:") && from == addr {
                    return Ok((local_addr, addr));
                }
            }
        }
        
        Err(AIError::NetworkError("Direct connection failed".to_string()))
    }
    
    /// Try hole punching.
    async fn try_hole_punching(
        &self,
        peer_id: &str,
        candidates: &[ConnectionCandidate],
    ) -> Result<(SocketAddr, SocketAddr)> {
        // Get local address
        let local_addr = self.socket.local_addr()
            .map_err(|e| AIError::NetworkError(format!("Failed to get local address: {}", e)))?;
        
        // Try each server-reflexive candidate
        for candidate in candidates {
            // Skip non-srflx candidates
            if candidate.candidate_type != "srflx" {
                continue;
            }
            
            // Parse address
            let addr = format!("{}:{}", candidate.ip, candidate.port).parse::<SocketAddr>()
                .map_err(|e| AIError::NetworkError(format!("Failed to parse address: {}", e)))?;
            
            // Send multiple pings to punch a hole
            let ping = format!("PING:{}", peer_id);
            
            for _ in 0..5 {
                self.socket.send_to(ping.as_bytes(), addr).await
                    .map_err(|e| AIError::NetworkError(format!("Failed to send ping: {}", e)))?;
                
                time::sleep(Duration::from_millis(100)).await;
            }
            
            // Wait for pong
            let mut buf = [0u8; 1024];
            let timeout = time::timeout(
                self.config.hole_punching_timeout,
                self.socket.recv_from(&mut buf)
            ).await;
            
            if let Ok(Ok((len, from))) = timeout {
                let message = String::from_utf8_lossy(&buf[..len]);
                if message.starts_with("PONG:") {
                    return Ok((local_addr, from));
                }
            }
        }
        
        Err(AIError::NetworkError("Hole punching failed".to_string()))
    }
    
    /// Try TURN relay.
    async fn try_turn_relay(
        &self,
        peer_id: &str,
        candidates: &[ConnectionCandidate],
    ) -> Result<(SocketAddr, SocketAddr)> {
        // Check if we have any TURN allocations
        if self.turn_allocations.read().is_empty() {
            // Create a TURN allocation
            self.create_turn_allocation().await?;
        }
        
        // Get local address
        let local_addr = self.socket.local_addr()
            .map_err(|e| AIError::NetworkError(format!("Failed to get local address: {}", e)))?;
        
        // Get TURN allocation
        let allocation = self.turn_allocations.read().values().next()
            .ok_or_else(|| AIError::NetworkError("No TURN allocation available".to_string()))?
            .clone();
        
        // Try each candidate
        for candidate in candidates {
            // Parse address
            let addr = format!("{}:{}", candidate.ip, candidate.port).parse::<SocketAddr>()
                .map_err(|e| AIError::NetworkError(format!("Failed to parse address: {}", e)))?;
            
            // Send ping through TURN
            let ping = format!("PING:{}", peer_id);
            
            // In a real implementation, this would use the TURN protocol
            // For now, just simulate it
            
            // Wait for pong
            let mut buf = [0u8; 1024];
            let timeout = time::timeout(Duration::from_secs(5), self.socket.recv_from(&mut buf)).await;
            
            if let Ok(Ok((len, from))) = timeout {
                let message = String::from_utf8_lossy(&buf[..len]);
                if message.starts_with("PONG:") {
                    return Ok((allocation.relayed_address, addr));
                }
            }
        }
        
        Err(AIError::NetworkError("TURN relay failed".to_string()))
    }
    
    /// Create a TURN allocation.
    ///
    /// # Returns
    ///
    /// A Result containing the TURN allocation or an error
    pub async fn create_turn_allocation(&self) -> Result<TURNAllocation> {
        // Check if we have any TURN servers
        if self.config.turn_servers.is_empty() {
            return Err(AIError::NetworkError("No TURN servers configured".to_string()));
        }
        
        // Use the first TURN server
        let (url, username, credential) = &self.config.turn_servers[0];
        
        // Parse TURN server address
        let server_addr = parse_turn_server(url)?;
        
        // In a real implementation, this would use the TURN protocol
        // For now, just create a dummy allocation
        
        let allocation = TURNAllocation {
            id: Uuid::new_v4().to_string(),
            relayed_address: SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::new(192, 0, 2, 1)), 12345),
            server_address: server_addr,
            username: username.clone(),
            password: credential.clone(),
            created_at: Instant::now(),
            expires_at: Instant::now() + self.config.turn_allocation_lifetime,
        };
        
        // Notify about allocation
        self.event_sender.send(NATTraversalEvent::TURNAllocationCreated(allocation.clone()))
            .map_err(|e| AIError::NetworkError(format!("Failed to send event: {}", e)))?;
        
        Ok(allocation)
    }
}

/// Parse a STUN server address.
fn parse_stun_server(server: &str) -> Result<SocketAddr> {
    // Remove stun: prefix
    let server = server.trim_start_matches("stun:");
    
    // Parse address
    server.parse::<SocketAddr>()
        .map_err(|e| AIError::NetworkError(format!("Failed to parse STUN server address: {}", e)))
}

/// Parse a TURN server address.
fn parse_turn_server(server: &str) -> Result<SocketAddr> {
    // Remove turn: prefix
    let server = server.trim_start_matches("turn:");
    
    // Parse address
    server.parse::<SocketAddr>()
        .map_err(|e| AIError::NetworkError(format!("Failed to parse TURN server address: {}", e)))
}

/// Create a STUN binding request.
fn create_stun_binding_request() -> Vec<u8> {
    // This is a simplified implementation for demonstration purposes
    // In a real implementation, this would create a proper STUN message
    
    let mut request = Vec::new();
    
    // Message type: Binding Request
    request.push(0x00);
    request.push(0x01);
    
    // Message length: 0
    request.push(0x00);
    request.push(0x00);
    
    // Magic cookie
    request.push(0x21);
    request.push(0x12);
    request.push(0xA4);
    request.push(0x42);
    
    // Transaction ID (12 bytes)
    for _ in 0..12 {
        request.push(rand::random::<u8>());
    }
    
    request
}

/// Parse a STUN response.
fn parse_stun_response(response: &[u8]) -> Result<STUNMessage> {
    // This is a simplified implementation for demonstration purposes
    // In a real implementation, this would properly parse the STUN message
    
    if response.len() < 20 {
        return Err(AIError::NetworkError("STUN response too short".to_string()));
    }
    
    // Extract message type
    let message_type = match (response[0], response[1]) {
        (0x01, 0x01) => STUNMessageType::BindingResponse,
        (0x01, 0x11) => STUNMessageType::BindingErrorResponse,
        _ => return Err(AIError::NetworkError("Unknown STUN message type".to_string())),
    };
    
    // Extract transaction ID
    let mut transaction_id = [0u8; 12];
    transaction_id.copy_from_slice(&response[8..20]);
    
    // Create message
    let mut message = STUNMessage {
        message_type,
        transaction_id,
        mapped_address: None,
        changed_address: None,
        source_address: None,
        changed_port: None,
        error_code: None,
        error_message: None,
    };
    
    // In a real implementation, we would parse attributes here
    // For now, just create a dummy mapped address
    message.mapped_address = Some(SocketAddr::new(
        IpAddr::V4(std::net::Ipv4Addr::new(203, 0, 113, 5)),
        12345,
    ));
    
    Ok(message)
}

/// Get the local IP address.
fn get_local_ip() -> Result<IpAddr> {
    // This is a simplified implementation for demonstration purposes
    // In a real implementation, this would get the actual local IP
    
    Ok(IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 2)))
}
