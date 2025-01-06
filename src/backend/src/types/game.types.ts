import { CRDTDocument, CRDTChange } from '../types/crdt.types';
import * as Automerge from 'automerge'; // v2.0

/**
 * Constants for game configuration and limits
 */
export const MAX_PLAYERS = 32;
export const MIN_SCAN_QUALITY = 0.6;
export const PHYSICS_UPDATE_RATE = 60;
export const STATE_SYNC_INTERVAL = 50;

/**
 * Enum for game state lifecycle management
 */
export enum GameStateType {
    INITIALIZING = 'INITIALIZING',
    RUNNING = 'RUNNING',
    PAUSED = 'PAUSED',
    TERMINATED = 'TERMINATED'
}

/**
 * Enum for sync priority levels
 */
export enum SyncPriority {
    HIGH = 'HIGH',
    MEDIUM = 'MEDIUM',
    LOW = 'LOW'
}

/**
 * Interface for 3D bounding box
 */
export interface BoundingBox3D {
    min: Vector3D;
    max: Vector3D;
}

/**
 * Interface for 3D vector
 */
export interface Vector3D {
    x: number;
    y: number;
    z: number;
}

/**
 * Interface for classified object from LiDAR
 */
export interface ClassifiedObject {
    id: string;
    type: string;
    confidence: number;
    boundingBox: BoundingBox3D;
    pointCount: number;
}

/**
 * Interface for mesh data
 */
export interface MeshData {
    vertices: Float32Array;
    indices: Uint32Array;
    normals: Float32Array;
    timestamp: number;
    quality: number;
}

/**
 * Interface for scanner health metrics
 */
export interface ScannerHealthMetrics {
    temperature: number;
    errorRate: number;
    uptime: number;
    lastCalibration: number;
}

/**
 * Interface for physics object
 */
export interface PhysicsObject {
    id: string;
    position: Vector3D;
    velocity: Vector3D;
    acceleration: Vector3D;
    mass: number;
    isStatic: boolean;
    collisionMask: number;
}

/**
 * Interface for collision event
 */
export interface CollisionEvent {
    objectA: string;
    objectB: string;
    point: Vector3D;
    normal: Vector3D;
    impulse: number;
    timestamp: number;
}

/**
 * Interface for force field
 */
export interface ForceField {
    position: Vector3D;
    radius: number;
    strength: number;
    type: 'ATTRACT' | 'REPEL';
}

/**
 * Interface for physics constraint
 */
export interface PhysicsConstraint {
    type: 'DISTANCE' | 'HINGE' | 'POINT';
    objectA: string;
    objectB: string;
    parameters: Record<string, number>;
}

/**
 * Type for environment state including LiDAR data
 */
export type EnvironmentState = {
    scanQuality: number;
    pointCount: number;
    classifiedObjects: ClassifiedObject[];
    meshData: MeshData;
    lastUpdateTimestamp: number;
    scannerHealth: ScannerHealthMetrics;
    environmentBounds: BoundingBox3D;
};

/**
 * Type for physics simulation state
 */
export type PhysicsState = {
    objects: PhysicsObject[];
    collisions: CollisionEvent[];
    timestamp: number;
    deltaTime: number;
    forceFields: ForceField[];
    constraints: PhysicsConstraint[];
};

/**
 * Type for game synchronization event
 */
export type GameSyncEvent = {
    type: GameStateType;
    changes: CRDTChange[];
    timestamp: number;
    latency: number;
    retryCount: number;
    priority: SyncPriority;
};

/**
 * Interface for game configuration
 */
export interface GameConfig {
    maxPlayers: number;
    scanQualityThreshold: number;
    physicsUpdateRate: number;
    syncInterval: number;
}

/**
 * Interface for game performance metrics
 */
export interface GamePerformanceMetrics {
    fps: number;
    frameTime: number;
    physicsLatency: number;
    renderLatency: number;
    networkLatency: number;
    memoryUsage: number;
}

/**
 * Interface for game session state
 */
export interface GameSessionState extends CRDTDocument {
    config: GameConfig;
    environment: EnvironmentState;
    physics: PhysicsState;
    performance: GamePerformanceMetrics;
    players: Map<string, PlayerState>;
    lastUpdateTimestamp: number;
}

/**
 * Interface for player state
 */
export interface PlayerState {
    id: string;
    position: Vector3D;
    rotation: Vector3D;
    health: number;
    latency: number;
    lastUpdateTimestamp: number;
}

/**
 * Type for game state document with Automerge integration
 */
export type GameStateDocument = Automerge.Doc<GameSessionState>;

/**
 * Interface for game event
 */
export interface GameEvent {
    type: string;
    timestamp: number;
    source: string;
    data: unknown;
    priority: SyncPriority;
}

/**
 * Interface for game update batch
 */
export interface GameUpdateBatch {
    events: GameEvent[];
    timestamp: number;
    batchId: string;
    priority: SyncPriority;
}