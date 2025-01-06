// External imports - WebRTC M98
import type { RTCPeerConnection, RTCDataChannel } from 'webrtc';

// Constants for fleet configuration and constraints
export const MAX_FLEET_SIZE = 32;
export const DEFAULT_SYNC_INTERVAL = 50; // milliseconds
export const MAX_SYNC_RETRIES = 3;
export const SYNC_TIMEOUT = 1000; // milliseconds
export const MIN_LEADER_SCORE = 0.8;
export const MAX_LATENCY_THRESHOLD = 50; // milliseconds

/**
 * Enum defining possible fleet operational states
 */
export enum FleetStatus {
    ACTIVE = 'ACTIVE',
    INACTIVE = 'INACTIVE',
    CONNECTING = 'CONNECTING',
    DEGRADED = 'DEGRADED'
}

/**
 * Enum defining possible roles within a fleet with leadership hierarchy
 */
export enum FleetRole {
    LEADER = 'LEADER',
    MEMBER = 'MEMBER',
    BACKUP_LEADER = 'BACKUP_LEADER'
}

/**
 * Enum defining types of messages exchanged between fleet members
 */
export enum FleetMessageType {
    STATE_UPDATE = 'STATE_UPDATE',
    MEMBER_JOIN = 'MEMBER_JOIN',
    MEMBER_LEAVE = 'MEMBER_LEAVE',
    PING = 'PING',
    CRDT_OPERATION = 'CRDT_OPERATION',
    LEADER_ELECTION = 'LEADER_ELECTION'
}

/**
 * Interface for comprehensive fleet network performance metrics
 */
export interface FleetNetworkStats {
    averageLatency: number;  // milliseconds
    maxLatency: number;      // milliseconds
    minLatency: number;      // milliseconds
    packetsLost: number;     // count
    bandwidth: number;       // bytes/second
    connectedPeers: number;  // count
    syncLatency: number;     // milliseconds
}

/**
 * Interface for WebRTC connection details and quality metrics of fleet members
 */
export interface FleetMemberConnection {
    peerConnection: RTCPeerConnection;
    dataChannel: RTCDataChannel;
    lastPing: number;         // timestamp
    connectionQuality: number; // 0-1 score
    retryCount: number;       // connection retry attempts
}

/**
 * Interface for structured messages exchanged during fleet synchronization
 */
export interface FleetSyncMessage {
    type: FleetMessageType;
    payload: any;
    timestamp: number;
    senderId: string;
    sequence: number;    // message sequence for ordering
    priority: number;    // message priority (0-9)
}

/**
 * Interface for tracking individual fleet member performance metrics
 */
export interface FleetMemberStats {
    latency: number;       // milliseconds
    packetsLost: number;   // count
    lastPing: number;      // timestamp
    bandwidth: number;     // bytes/second
    connectionScore: number; // 0-1 score
    syncSuccess: number;    // percentage
    leaderScore: number;    // 0-1 score for leader election
}

/**
 * Type guard for checking valid fleet size
 */
export function isValidFleetSize(size: number): boolean {
    return size > 0 && size <= MAX_FLEET_SIZE;
}

/**
 * Type guard for checking valid latency threshold
 */
export function isValidLatency(latency: number): boolean {
    return latency >= 0 && latency <= MAX_LATENCY_THRESHOLD;
}