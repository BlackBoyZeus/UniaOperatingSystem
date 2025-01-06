import { z } from 'zod';
import type { 
    RTCPeerConnection, 
    RTCDataChannel 
} from 'webrtc'; // WebRTC M98
import {
    FleetStatus,
    FleetRole,
    FleetMessageType,
    FleetNetworkStats,
    FleetMemberConnection,
    FleetSyncMessage,
    FleetQualityMetrics,
    FleetCRDTOperation,
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD
} from '../types/fleet.types';

/**
 * Core fleet interface with enhanced monitoring and redundancy capabilities
 * @interface IFleet
 */
export interface IFleet {
    /** Unique identifier for the fleet */
    id: string;
    
    /** Human-readable fleet name */
    name: string;
    
    /** Maximum number of devices allowed (up to 32) */
    maxDevices: number;
    
    /** Array of current fleet members */
    members: IFleetMember[];
    
    /** Current operational status of the fleet */
    status: FleetStatus;
    
    /** Comprehensive network performance metrics */
    networkStats: FleetNetworkStats;
    
    /** Fleet-wide quality metrics */
    qualityMetrics: FleetQualityMetrics;
    
    /** Ordered list of backup leader IDs for redundancy */
    backupLeaders: string[];
}

/**
 * Enhanced fleet member interface with quality metrics and CRDT support
 * @interface IFleetMember
 */
export interface IFleetMember {
    /** Unique member identifier */
    id: string;
    
    /** Associated device identifier */
    deviceId: string;
    
    /** Member's role in the fleet hierarchy */
    role: FleetRole;
    
    /** WebRTC connection details */
    connection: FleetMemberConnection;
    
    /** Current P2P latency in milliseconds */
    latency: number;
    
    /** Individual connection quality metrics */
    connectionQuality: FleetQualityMetrics;
    
    /** Last CRDT operation details */
    lastCRDTOperation: FleetCRDTOperation;
}

/**
 * Enhanced WebRTC connection interface with quality monitoring
 * @interface IFleetConnection
 */
export interface IFleetConnection {
    /** WebRTC peer connection instance */
    peerConnection: RTCPeerConnection;
    
    /** Data channel for P2P communication */
    dataChannel: RTCDataChannel;
    
    /** Handler for fleet sync messages */
    messageHandler: (message: FleetSyncMessage) => Promise<void>;
    
    /** Quality metrics monitoring callback */
    qualityMonitor: (metrics: FleetQualityMetrics) => void;
}

/**
 * Zod schema for runtime fleet data validation
 */
export const IFleetSchema = z.object({
    id: z.string().uuid(),
    name: z.string().min(1).max(64),
    maxDevices: z.number().min(1).max(MAX_FLEET_SIZE),
    members: z.array(z.lazy(() => IFleetMemberSchema)),
    status: z.nativeEnum(FleetStatus),
    networkStats: z.object({
        averageLatency: z.number().min(0).max(MAX_LATENCY_THRESHOLD),
        maxLatency: z.number().min(0),
        minLatency: z.number().min(0),
        packetsLost: z.number().min(0),
        bandwidth: z.number().min(0),
        connectedPeers: z.number().min(0).max(MAX_FLEET_SIZE),
        syncLatency: z.number().min(0)
    }),
    qualityMetrics: z.object({
        connectionScore: z.number().min(0).max(1),
        syncSuccess: z.number().min(0).max(100),
        leaderRedundancy: z.number().min(0).max(1)
    }),
    backupLeaders: z.array(z.string().uuid())
});

/**
 * Zod schema for runtime fleet member validation
 */
export const IFleetMemberSchema = z.object({
    id: z.string().uuid(),
    deviceId: z.string(),
    role: z.nativeEnum(FleetRole),
    connection: z.object({
        lastPing: z.number(),
        connectionQuality: z.number().min(0).max(1),
        retryCount: z.number().min(0)
    }),
    latency: z.number().min(0).max(MAX_LATENCY_THRESHOLD),
    connectionQuality: z.object({
        signalStrength: z.number().min(0).max(1),
        stability: z.number().min(0).max(1),
        reliability: z.number().min(0).max(1)
    }),
    lastCRDTOperation: z.object({
        timestamp: z.number(),
        type: z.string(),
        payload: z.any()
    })
});