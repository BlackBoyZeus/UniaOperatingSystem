import { RTCPeerConnection, RTCDataChannel, RTCConfiguration } from 'webrtc'; // version: M98
import { IPointCloud } from '../interfaces/lidar.interface';

/**
 * Maximum number of peers supported in a mesh network fleet
 */
export const MAX_PEERS = 32;

/**
 * Maximum acceptable P2P network latency in milliseconds
 */
export const MAX_LATENCY = 50;

/**
 * Timeout duration for peer reconnection attempts in milliseconds
 */
export const RECONNECT_TIMEOUT = 5000;

/**
 * Timeout duration for ICE candidate gathering in milliseconds
 */
export const ICE_GATHERING_TIMEOUT = 5000;

/**
 * Configuration interface for mesh network settings
 */
export interface MeshConfig {
    maxPeers: number;                  // Maximum number of peers in fleet (≤32)
    maxLatency: number;                // Maximum acceptable latency in ms (≤50)
    iceServers: RTCIceServer[];        // WebRTC ICE server configuration
    reconnectTimeout?: number;         // Custom reconnection timeout
    gatheringTimeout?: number;         // Custom ICE gathering timeout
}

/**
 * Interface for mesh network peer information
 */
export interface MeshPeer {
    id: string;                        // Unique peer identifier
    connection: RTCPeerConnection;     // WebRTC peer connection
    dataChannel: RTCDataChannel;       // WebRTC data channel for P2P communication
    latency: number;                   // Current P2P latency in milliseconds
    lastSeen?: number;                 // Timestamp of last activity
    connectionState: RTCPeerConnectionState;
    environmentData?: IPointCloud;     // Latest shared environment data
}

/**
 * Interface defining mesh network topology structure
 */
export interface MeshTopology {
    peers: Map<string, MeshPeer>;              // Active peer connections
    connections: Map<string, string[]>;        // Peer connection graph
    environmentData: Map<string, IPointCloud>; // Shared environment data
    fleetId?: string;                         // Unique fleet identifier
    createdAt: number;                        // Fleet creation timestamp
    lastUpdated: number;                      // Last topology update timestamp
}

/**
 * Enum for mesh network event types
 */
export enum MeshEventType {
    PEER_CONNECTED = 'PEER_CONNECTED',
    PEER_DISCONNECTED = 'PEER_DISCONNECTED',
    TOPOLOGY_UPDATED = 'TOPOLOGY_UPDATED',
    LATENCY_CHANGED = 'LATENCY_CHANGED',
    ENVIRONMENT_UPDATED = 'ENVIRONMENT_UPDATED',
    FLEET_FULL = 'FLEET_FULL',
    FLEET_DISSOLVED = 'FLEET_DISSOLVED'
}

/**
 * Interface for mesh network events
 */
export interface MeshEvent {
    type: MeshEventType;              // Event type identifier
    peerId: string;                   // Affected peer ID
    data: any;                        // Event-specific data
    timestamp: number;                // Event timestamp
    fleetId?: string;                // Associated fleet ID
}

/**
 * Type guard to validate peer count against MAX_PEERS
 * @param count Number of peers to validate
 * @returns Whether peer count is valid
 */
export function isValidPeerCount(count: number): boolean {
    return typeof count === 'number' && count > 0 && count <= MAX_PEERS;
}

/**
 * Type guard to validate network latency
 * @param latency Latency value to validate
 * @returns Whether latency meets requirements
 */
export function isValidLatency(latency: number): boolean {
    return typeof latency === 'number' && latency > 0 && latency <= MAX_LATENCY;
}

/**
 * Interface for WebRTC connection statistics
 */
export interface ConnectionStats {
    bytesReceived: number;
    bytesSent: number;
    packetsReceived: number;
    packetsSent: number;
    packetsLost: number;
    roundTripTime: number;
    timestamp: number;
}

/**
 * Interface for mesh network health metrics
 */
export interface MeshHealthMetrics {
    averageLatency: number;
    maxLatency: number;
    minLatency: number;
    packetLossRate: number;
    connectionStability: number;
    lastUpdated: number;
}

/**
 * Type for CRDT-based state synchronization
 */
export type CRDTState = {
    vector: Map<string, number>;      // Vector clock
    values: Map<string, any>;         // Replicated values
    timestamp: number;                // Last update timestamp
};