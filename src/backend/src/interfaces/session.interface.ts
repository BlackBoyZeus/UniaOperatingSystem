/**
 * @file Session interface definitions for TALD UNIA platform
 * @version 1.0.0
 */

import { IGameState } from '../types/game.types';
import { IFleetMember, FleetConfig, FleetMemberStats } from '../types/fleet.types';
import { PointCloudData } from '../types/lidar.types';
import { RTCPeerConnection } from 'webrtc'; // v.M98

// Global constants for session management
export const MAX_SESSION_PARTICIPANTS = 32; // Maximum fleet size
export const DEFAULT_SCAN_RATE = 30; // 30Hz scan rate
export const MAX_SESSION_DURATION = 14400000; // 4 hours in milliseconds
export const MAX_LATENCY_THRESHOLD = 50; // 50ms maximum P2P latency
export const PERFORMANCE_CHECK_INTERVAL = 1000; // 1 second performance check interval
export const STATE_SYNC_INTERVAL = 50; // 50ms state sync interval

/**
 * Enum defining possible session states including performance states
 */
export enum SessionStatus {
    INITIALIZING = 'initializing',
    ACTIVE = 'active',
    PAUSED = 'paused',
    TERMINATED = 'terminated',
    DEGRADED = 'degraded'
}

/**
 * Interface for tracking detailed session performance metrics
 */
export interface IPerformanceMetrics {
    averageLatency: number;
    packetLoss: number;
    syncRate: number;
    participantMetrics: Map<string, FleetMemberStats>;
    cpuUsage: number;
    memoryUsage: number;
    batteryLevel: number;
    networkBandwidth: number;
    scanQuality: number;
    frameRate: number;
    lastUpdate: number;
}

/**
 * Interface defining performance thresholds for session management
 */
export interface IPerformanceThresholds {
    maxLatency: number;
    maxPacketLoss: number;
    minSyncRate: number;
    minBatteryLevel: number;
    minFrameRate: number;
    minScanQuality: number;
    maxCpuUsage: number;
    maxMemoryUsage: number;
}

/**
 * Interface for WebRTC peer connection state
 */
export interface IPeerConnection {
    connection: RTCPeerConnection;
    participantId: string;
    latency: number;
    lastPing: number;
    status: 'connected' | 'connecting' | 'disconnected';
}

/**
 * Interface for session configuration
 */
export interface ISessionConfig {
    maxParticipants: number;
    networkConfig: FleetConfig;
    scanRate: number;
    performanceThresholds: IPerformanceThresholds;
    autoRecoveryEnabled: boolean;
    meshTopology: 'full' | 'star' | 'ring';
    stateValidation: boolean;
    compressionEnabled: boolean;
    encryptionEnabled: boolean;
}

/**
 * Interface for session state monitoring
 */
export interface ISessionState {
    status: SessionStatus;
    activeParticipants: number;
    averageLatency: number;
    lastUpdate: Date;
    performanceMetrics: IPerformanceMetrics;
    environmentData: PointCloudData;
    peerConnections: Map<string, IPeerConnection>;
    errorCount: number;
    warningCount: number;
    recoveryAttempts: number;
}

/**
 * Interface for session error tracking
 */
export interface ISessionError {
    code: string;
    message: string;
    severity: 'critical' | 'warning' | 'info';
    timestamp: Date;
    participantId?: string;
    recoveryAction?: string;
}

/**
 * Main session interface with comprehensive management capabilities
 */
export interface ISession {
    sessionId: string;
    startTime: Date;
    participants: IFleetMember[];
    gameState: IGameState;
    config: ISessionConfig;
    state: ISessionState;
    performance: IPerformanceMetrics;
    errors: ISessionError[];
    lastStateSync: number;
    recoveryMode: boolean;

    // Session lifecycle methods
    initialize(): Promise<void>;
    terminate(): Promise<void>;
    pause(): Promise<void>;
    resume(): Promise<void>;

    // Participant management
    addParticipant(participant: IFleetMember): Promise<boolean>;
    removeParticipant(participantId: string): Promise<void>;
    validateParticipant(participantId: string): boolean;

    // State management
    syncState(): Promise<void>;
    validateState(): boolean;
    rollbackState(timestamp: number): Promise<boolean>;

    // Performance monitoring
    checkPerformance(): IPerformanceMetrics;
    optimizePerformance(): Promise<void>;
    handleDegradedPerformance(): Promise<void>;

    // Error handling
    logError(error: ISessionError): void;
    attemptRecovery(): Promise<boolean>;
    generateDiagnostics(): Record<string, any>;
}

/**
 * Interface for session factory creation
 */
export interface ISessionFactory {
    createSession(config: ISessionConfig): Promise<ISession>;
    validateConfig(config: ISessionConfig): boolean;
    getActiveSessions(): Map<string, ISession>;
}