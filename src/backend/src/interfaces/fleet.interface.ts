import { CRDTDocument, CRDTChange, CRDTOperation } from '../types/crdt.types';
import { RTCPeerConnection, RTCDataChannel } from 'webrtc'; // v.M98
import { Doc } from 'automerge'; // v2.0

/**
 * Core fleet interface defining fleet properties and management capabilities
 */
export interface IFleet {
    id: string;
    name: string;
    maxDevices: number; // Maximum 32 devices per fleet
    members: IFleetMember[];
    state: IFleetState;
    meshConfig: IMeshConfig;
    networkStats: IFleetNetworkStats;
    securityConfig: IFleetSecurity;
    createdAt: number;
    lastUpdated: number;
}

/**
 * Interface for individual fleet members
 */
export interface IFleetMember {
    id: string;
    deviceId: string;
    role: FleetRole;
    status: FleetStatus;
    joinedAt: number;
    lastActive: number;
    peerConnection: RTCPeerConnection;
    dataChannel: RTCDataChannel;
    position: IPosition;
    capabilities: IDeviceCapabilities;
}

/**
 * Interface for 3D position tracking
 */
export interface IPosition {
    x: number;
    y: number;
    z: number;
    timestamp: number;
    accuracy: number;
}

/**
 * Interface for device capabilities
 */
export interface IDeviceCapabilities {
    lidarSupport: boolean;
    maxRange: number;
    processingPower: number;
    networkBandwidth: number;
    batteryLevel: number;
}

/**
 * Interface for fleet state management using CRDT
 */
export interface IFleetState extends CRDTDocument {
    gameState: Doc<any>;
    environmentState: Doc<any>;
    syncTimestamp: number;
    stateVersion: number;
    pendingChanges: CRDTChange[];
    lastMergeTimestamp: number;
}

/**
 * Interface for mesh network configuration
 */
export interface IMeshConfig {
    topology: MeshTopologyType;
    maxPeers: number;
    reconnectStrategy: IReconnectStrategy;
    peerTimeout: number;
    signalServer: string;
    iceServers: RTCIceServer[];
    meshQuality: IMeshQualityMetrics;
}

/**
 * Interface for mesh quality metrics
 */
export interface IMeshQualityMetrics {
    connectionDensity: number;
    redundancyFactor: number;
    meshStability: number;
    routingEfficiency: number;
}

/**
 * Interface for reconnection strategy
 */
export interface IReconnectStrategy {
    maxAttempts: number;
    backoffMultiplier: number;
    initialDelay: number;
    maxDelay: number;
}

/**
 * Interface for fleet network statistics
 */
export interface IFleetNetworkStats {
    averageLatency: number; // Must be ≤50ms
    peakLatency: number;
    packetLoss: number;
    bandwidth: IBandwidthStats;
    connectionQuality: number;
    meshHealth: number;
    lastUpdate: number;
}

/**
 * Interface for bandwidth statistics
 */
export interface IBandwidthStats {
    current: number;
    peak: number;
    average: number;
    totalTransferred: number;
    lastMeasured: number;
}

/**
 * Interface for fleet security configuration
 */
export interface IFleetSecurity {
    encryptionEnabled: boolean;
    authenticationMethod: FleetAuthMethod;
    accessControl: IFleetAccessControl;
    certificateConfig?: ICertificateConfig;
    tokenConfig?: ITokenConfig;
}

/**
 * Interface for fleet access control
 */
export interface IFleetAccessControl {
    allowedDevices: string[];
    bannedDevices: string[];
    joinPolicy: JoinPolicy;
    rolePermissions: Map<FleetRole, string[]>;
}

/**
 * Interface for certificate-based authentication
 */
export interface ICertificateConfig {
    issuer: string;
    validityPeriod: number;
    renewalThreshold: number;
    revocationList: string[];
}

/**
 * Interface for token-based authentication
 */
export interface ITokenConfig {
    validityDuration: number;
    refreshThreshold: number;
    maxRefreshes: number;
    blacklist: string[];
}

/**
 * Enum for fleet member roles
 */
export enum FleetRole {
    LEADER = 'leader',
    MEMBER = 'member',
    BACKUP_LEADER = 'backup_leader'
}

/**
 * Enum for fleet status
 */
export enum FleetStatus {
    ACTIVE = 'active',
    INACTIVE = 'inactive',
    CONNECTING = 'connecting',
    DEGRADED = 'degraded',
    RECOVERING = 'recovering'
}

/**
 * Enum for mesh peer status
 */
export enum MeshPeerStatus {
    CONNECTED = 'connected',
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    RECONNECTING = 'reconnecting',
    ERROR = 'error'
}

/**
 * Enum for mesh topology types
 */
export enum MeshTopologyType {
    FULL = 'full',
    STAR = 'star',
    RING = 'ring',
    HYBRID = 'hybrid'
}

/**
 * Enum for fleet authentication methods
 */
export enum FleetAuthMethod {
    TOKEN = 'token',
    CERTIFICATE = 'certificate',
    HARDWARE_ID = 'hardware_id'
}

/**
 * Enum for fleet join policies
 */
export enum JoinPolicy {
    OPEN = 'open',
    INVITE_ONLY = 'invite_only',
    APPROVAL_REQUIRED = 'approval_required',
    CLOSED = 'closed'
}