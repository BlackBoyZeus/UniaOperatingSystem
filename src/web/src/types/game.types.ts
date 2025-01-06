import { Point3D } from './lidar.types';
import { Automerge } from 'automerge'; // ^2.0.0

// Global constants for game performance targets
export const TARGET_FPS = 60 as const;
export const MIN_FPS = 30 as const;
export const MAX_PLAYERS = 32 as const;
export const MAX_FRAME_TIME = 16.67 as const;
export const MAX_LIDAR_LATENCY = 50 as const;
export const MAX_NETWORK_LATENCY = 50 as const;

/**
 * Enum defining possible game states
 */
export enum GameState {
    INITIALIZING = 'INITIALIZING',
    LOADING = 'LOADING',
    RUNNING = 'RUNNING',
    PAUSED = 'PAUSED',
    ENDED = 'ENDED'
}

/**
 * Enum defining render quality levels
 */
export enum RenderQuality {
    LOW = 'LOW',
    MEDIUM = 'MEDIUM',
    HIGH = 'HIGH'
}

/**
 * Type for tracking real-time performance metrics
 */
export type PerformanceMetrics = {
    readonly fps: number;
    readonly frameTime: number;
    readonly lidarLatency: number;
    readonly networkLatency: number;
    readonly memoryUsage: number;
};

/**
 * Enhanced type for AI-classified objects with confidence scores and bounding boxes
 */
export type ClassifiedObject = {
    readonly id: string;
    readonly type: string;
    readonly position: Point3D;
    readonly confidence: number;
    readonly boundingBox: {
        readonly min: Point3D;
        readonly max: Point3D;
    };
};

/**
 * Enhanced type for game environment state with versioning and timestamps
 */
export type EnvironmentState = {
    readonly meshData: ArrayBuffer;
    readonly pointCloud: Float32Array;
    readonly classifiedObjects: readonly ClassifiedObject[];
    readonly timestamp: number;
    readonly version: string;
};

/**
 * Enhanced type for render configuration with additional graphics options
 */
export type RenderConfig = {
    readonly resolution: {
        readonly width: number;
        readonly height: number;
    };
    readonly quality: RenderQuality;
    readonly lidarOverlayEnabled: boolean;
    readonly vsyncEnabled: boolean;
    readonly antiAliasing: number;
};

/**
 * Enhanced type for complete game state data with performance tracking and fleet management
 * Implements CRDT support via Automerge for state synchronization
 */
export type GameStateData = Automerge.Doc<{
    readonly gameId: string;
    readonly sessionId: string;
    readonly state: GameState;
    readonly environment: EnvironmentState;
    readonly renderConfig: RenderConfig;
    readonly performance: PerformanceMetrics;
    readonly fleetSize: number;
    readonly lastSync: number;
}>;

/**
 * Type guard for validating performance metrics against target thresholds
 */
export function isPerformanceAcceptable(metrics: PerformanceMetrics): boolean {
    return (
        metrics.fps >= MIN_FPS &&
        metrics.frameTime <= MAX_FRAME_TIME &&
        metrics.lidarLatency <= MAX_LIDAR_LATENCY &&
        metrics.networkLatency <= MAX_NETWORK_LATENCY
    );
}

/**
 * Type guard for validating fleet size against maximum players limit
 */
export function isFleetSizeValid(size: number): boolean {
    return size >= 1 && size <= MAX_PLAYERS;
}

/**
 * Type guard for validating environment state freshness
 */
export function isEnvironmentStateFresh(state: EnvironmentState): boolean {
    return Date.now() - state.timestamp <= MAX_LIDAR_LATENCY;
}