import { Point3D } from '../types/lidar.types';

/**
 * Enumeration of possible game states with strict type safety
 */
export enum GameStates {
    INITIALIZING = 'INITIALIZING',
    LOADING = 'LOADING',
    RUNNING = 'RUNNING',
    PAUSED = 'PAUSED',
    ENDED = 'ENDED',
    ERROR = 'ERROR'
}

/**
 * Enumeration of core game events for state management
 */
export enum GameEvents {
    START = 'START',
    PAUSE = 'PAUSE',
    RESUME = 'RESUME',
    END = 'END'
}

/**
 * Render quality levels for graphics configuration
 */
export enum RenderQuality {
    LOW = 'LOW',
    MEDIUM = 'MEDIUM',
    HIGH = 'HIGH'
}

/**
 * Interface for AI-classified objects in the web environment
 * Ensures immutability for thread safety
 */
export interface IWebClassifiedObject {
    readonly id: string;
    readonly type: string;
    readonly position: Point3D;
}

/**
 * Interface for web-specific environment state
 * Includes LiDAR data and object classification
 */
export interface IWebEnvironmentState {
    readonly meshData: ArrayBuffer;
    readonly pointCloud: Float32Array;
    readonly classifiedObjects: readonly IWebClassifiedObject[];
    readonly timestamp: number;
}

/**
 * Interface for web render configuration
 * Supports dynamic quality adjustment and LiDAR overlay
 */
export interface IWebRenderState {
    readonly resolution: {
        readonly width: number;
        readonly height: number;
    };
    readonly quality: RenderQuality;
    readonly lidarOverlayEnabled: boolean;
}

/**
 * Branded type for FPS to ensure valid frame rate values
 */
type ValidFPS = number & { _brand: 'ValidFPS' };

/**
 * Core interface for web-specific game state
 * Implements comprehensive state management with strict null checks
 */
export interface IWebGameState {
    readonly gameId: string;
    readonly sessionId: string;
    readonly state: GameStates;
    readonly environmentData: IWebEnvironmentState | null;
    readonly renderState: IWebRenderState;
    readonly fps: ValidFPS;
}