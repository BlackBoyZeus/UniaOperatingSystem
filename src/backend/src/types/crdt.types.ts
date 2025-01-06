// @ts-nocheck
import { Doc } from 'automerge'; // v2.0 - CRDT implementation library

/**
 * Constants for CRDT synchronization configuration
 */
export const DEFAULT_SYNC_INTERVAL = 50; // 50ms sync interval for real-time updates
export const MAX_RETRIES = 3; // Maximum retry attempts for failed operations
export const SYNC_TIMEOUT = 1000; // 1 second timeout for sync operations
export const MAX_FLEET_SIZE = 32; // Maximum supported fleet size
export const MAX_LATENCY_THRESHOLD = 50; // Maximum acceptable latency in ms

/**
 * Enum defining supported CRDT operations
 */
export enum CRDTOperation {
    INSERT = 'INSERT',
    UPDATE = 'UPDATE',
    DELETE = 'DELETE',
    MERGE = 'MERGE'
}

/**
 * Enum for exponential backoff strategies
 */
export enum BackoffStrategy {
    LINEAR = 'LINEAR',
    EXPONENTIAL = 'EXPONENTIAL',
    FIBONACCI = 'FIBONACCI'
}

/**
 * Base interface for CRDT document with performance metrics
 */
export interface CRDTDocument<T = any> {
    id: string;
    version: number;
    data: T;
    lastSyncTimestamp: number;
    syncLatency: number;
}

/**
 * Interface for CRDT change operations with retry tracking
 */
export interface CRDTChange {
    documentId: string;
    operation: CRDTOperation;
    timestamp: number;
    retryCount: number;
}

/**
 * Configuration interface for CRDT synchronization
 */
export interface CRDTSyncConfig {
    syncInterval: number;
    maxRetries: number;
    timeout: number;
    latencyThreshold: number;
    backoffStrategy: BackoffStrategy;
}

/**
 * Interface for fleet member status
 */
export interface FleetMemberStatus {
    connected: boolean;
    lastSeen: number;
    latency: number;
    syncErrors: number;
}

/**
 * Interface for fleet member state
 */
export interface FleetMemberState {
    id: string;
    status: FleetMemberStatus;
    position: {
        x: number;
        y: number;
        z: number;
    };
}

/**
 * Interface for fleet performance metrics
 */
export interface FleetPerformanceMetrics {
    averageLatency: number;
    syncSuccessRate: number;
    memberCount: number;
    lastUpdateTimestamp: number;
}

/**
 * Interface for environment state including LiDAR data
 */
export interface EnvironmentState {
    meshData: ArrayBuffer;
    pointCloud: Float32Array;
    boundingBox: {
        min: { x: number; y: number; z: number };
        max: { x: number; y: number; z: number };
    };
}

/**
 * Interface for physics state
 */
export interface PhysicsState {
    gravity: { x: number; y: number; z: number };
    collisions: Array<{
        objectA: string;
        objectB: string;
        point: { x: number; y: number; z: number };
        force: number;
    }>;
}

/**
 * Interface for LiDAR state
 */
export interface LiDARState {
    scanQuality: number;
    pointCount: number;
    scanTimestamp: number;
    processingLatency: number;
}

/**
 * Interface for game performance metrics
 */
export interface GamePerformanceMetrics {
    fps: number;
    frameTime: number;
    physicsUpdateTime: number;
    lidarProcessingTime: number;
}

/**
 * Enhanced CRDT document type for fleet state
 */
export type FleetCRDTDocument = CRDTDocument<{
    members: FleetMemberState[];
    status: FleetMemberStatus;
    performance: FleetPerformanceMetrics;
    maxSize: number;
}>;

/**
 * Enhanced CRDT document type for game state
 */
export type GameCRDTDocument = CRDTDocument<{
    environment: EnvironmentState;
    physics: PhysicsState;
    lidar: LiDARState;
    performance: GamePerformanceMetrics;
}>;

/**
 * Type for Automerge document wrapper
 */
export type AutomergeDoc<T> = Doc<T>;

/**
 * Interface for sync error details
 */
export interface SyncError {
    documentId: string;
    operation: CRDTOperation;
    timestamp: number;
    error: string;
    retryCount: number;
}

/**
 * Interface for sync statistics
 */
export interface SyncStats {
    totalOperations: number;
    successfulOperations: number;
    failedOperations: number;
    averageLatency: number;
    lastSyncTimestamp: number;
}