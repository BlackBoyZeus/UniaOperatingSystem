import { CRDTDocument, CRDTChange } from '../types/crdt.types';

/**
 * High-precision 3D vector representation with 0.01cm resolution support
 */
export interface Vector3 {
    x: number;  // X coordinate with 0.01cm precision
    y: number;  // Y coordinate with 0.01cm precision
    z: number;  // Z coordinate with 0.01cm precision
}

/**
 * Performance metrics interface for system latency monitoring
 * All latency values are in milliseconds
 */
export interface IPerformanceMetrics {
    stateUpdateLatency: number;     // Overall state update latency
    lidarProcessingLatency: number; // LiDAR processing pipeline latency
    physicsSimulationLatency: number; // Physics engine simulation latency
    fleetSyncLatency: number;       // Fleet synchronization latency
}

/**
 * LiDAR metrics interface for quality and performance monitoring
 */
export interface ILiDARMetrics {
    scanRate: number;      // Scan rate in Hz (target: 30Hz)
    resolution: number;    // Spatial resolution in cm (target: 0.01cm)
    effectiveRange: number; // Effective scanning range in meters
    pointDensity: number;  // Points per cubic meter
}

/**
 * Interface for classified objects detected by LiDAR
 */
interface IClassifiedObject {
    id: string;
    type: string;
    position: Vector3;
    dimensions: Vector3;
    confidence: number;    // Classification confidence score (0-1)
    timestamp: number;     // Detection timestamp
}

/**
 * Interface for physics objects in the simulation
 */
interface IPhysicsObject {
    id: string;
    position: Vector3;
    velocity: Vector3;
    mass: number;
    collisionMesh: ArrayBuffer;
    lastUpdateTimestamp: number;
}

/**
 * Interface for collision events in physics simulation
 */
interface ICollisionEvent {
    objectAId: string;
    objectBId: string;
    point: Vector3;
    force: number;
    timestamp: number;
}

/**
 * Environment state interface with LiDAR integration
 */
export interface IEnvironmentState {
    timestamp: number;                     // State timestamp
    scanQuality: number;                   // Overall scan quality (0-1)
    pointCount: number;                    // Total points in current scan
    classifiedObjects: IClassifiedObject[]; // Detected and classified objects
    lidarMetrics: ILiDARMetrics;           // Current LiDAR performance metrics
}

/**
 * Physics state interface with collision detection
 */
export interface IPhysicsState {
    timestamp: number;                // Physics state timestamp
    objects: IPhysicsObject[];       // Active physics objects
    collisions: ICollisionEvent[];   // Current frame collision events
    simulationLatency: number;       // Current simulation step latency
}

/**
 * Core game state interface with fleet management support
 * Implements CRDT-based synchronization for 32-device fleets
 */
export interface IGameState extends CRDTDocument {
    gameId: string;           // Unique game instance identifier
    sessionId: string;        // Current session identifier
    fleetId: string;         // Associated fleet identifier
    deviceCount: number;      // Current number of connected devices
    timestamp: number;        // State timestamp
    environment: IEnvironmentState; // Current environment state
    physics: IPhysicsState;   // Current physics state
    metrics: IPerformanceMetrics; // System performance metrics
}