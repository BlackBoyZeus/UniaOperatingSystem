import { throttle } from 'lodash'; // ^4.17.21
import * as Automerge from 'automerge'; // ^2.0.0
import { 
    GameState, 
    RenderQuality, 
    GameStateData, 
    EnvironmentState,
    isPerformanceAcceptable,
    isFleetSizeValid,
    isEnvironmentStateFresh
} from '../types/game.types';
import {
    IWebGameState,
    IWebEnvironmentState,
    IWebRenderState,
    IWebCRDTState
} from '../interfaces/game.interface';
import { LIDAR_PERFORMANCE } from '../constants/lidar.constants';

// Global constants for performance optimization
const TARGET_FPS = 60;
const MIN_FPS = 30;
const FPS_UPDATE_INTERVAL = 1000;
const STATE_UPDATE_THROTTLE = 16;
const POINT_CLOUD_OPTIMIZATION_THRESHOLD = 1_000_000;
const MESH_QUALITY_LEVELS = {
    LOW: 0.5,
    MEDIUM: 0.75,
    HIGH: 1.0
} as const;
const CRDT_SYNC_INTERVAL = 50;

// FPS calculation window for moving average
const FPS_WINDOW_SIZE = 10;
const fpsHistory: number[] = [];

/**
 * Enhanced FPS calculation with performance monitoring and adaptive throttling
 * @param currentTime - Current timestamp in milliseconds
 * @param lastFrameTime - Last frame timestamp in milliseconds
 * @param renderState - Current render state configuration
 * @returns Object containing calculated FPS and optimization flag
 */
export const calculateFPS = throttle((
    currentTime: number,
    lastFrameTime: number,
    renderState: IWebRenderState
): { fps: number; shouldOptimize: boolean } => {
    const frameDelta = currentTime - lastFrameTime;
    const instantFPS = 1000 / frameDelta;

    // Update FPS history with moving average
    fpsHistory.push(instantFPS);
    if (fpsHistory.length > FPS_WINDOW_SIZE) {
        fpsHistory.shift();
    }

    // Calculate weighted moving average
    const weightedFPS = fpsHistory.reduce((acc, fps, idx) => {
        const weight = (idx + 1) / fpsHistory.length;
        return acc + (fps * weight);
    }, 0) / fpsHistory.length;

    // Clamp FPS between MIN_FPS and TARGET_FPS
    const clampedFPS = Math.max(MIN_FPS, Math.min(TARGET_FPS, weightedFPS));

    // Determine if optimization is needed
    const shouldOptimize = clampedFPS < TARGET_FPS * 0.9 || // Below 90% of target
                          frameDelta > STATE_UPDATE_THROTTLE;

    return {
        fps: Math.round(clampedFPS),
        shouldOptimize
    };
}, FPS_UPDATE_INTERVAL);

/**
 * Enhanced game state validation with CRDT support and conflict resolution
 * @param gameState - Current game state
 * @param crdtState - CRDT state for synchronization
 * @returns Validation result and detected conflicts
 */
export const validateGameState = throttle((
    gameState: IWebGameState,
    crdtState: Automerge.Doc<GameStateData>
): { isValid: boolean; conflicts: string[] } => {
    const conflicts: string[] = [];

    // Validate base state structure
    if (!gameState.gameId || !gameState.sessionId) {
        conflicts.push('Invalid game or session ID');
    }

    // Validate game state enum
    if (!Object.values(GameState).includes(gameState.state)) {
        conflicts.push('Invalid game state value');
    }

    // Validate environment data if present
    if (gameState.environmentData) {
        if (!isEnvironmentStateFresh(gameState.environmentData as EnvironmentState)) {
            conflicts.push('Environment data outdated');
        }
    }

    // Validate render state
    if (!Object.values(RenderQuality).includes(gameState.renderState.quality)) {
        conflicts.push('Invalid render quality setting');
    }

    // Validate CRDT state consistency
    try {
        const changes = Automerge.getChanges(crdtState, Automerge.init());
        if (changes.length > 0) {
            const mergedState = Automerge.merge(crdtState, Automerge.load(changes));
            if (!isFleetSizeValid(mergedState.fleetSize)) {
                conflicts.push('Invalid fleet size in CRDT state');
            }
        }
    } catch (error) {
        conflicts.push(`CRDT validation error: ${error.message}`);
    }

    return {
        isValid: conflicts.length === 0,
        conflicts
    };
}, STATE_UPDATE_THROTTLE);

/**
 * Optimized environment data processing with enhanced point cloud handling
 * @param environmentData - Current environment state data
 * @param currentFPS - Current FPS for adaptive optimization
 * @returns Processed and optimized environment data
 */
export const processEnvironmentData = throttle((
    environmentData: IWebEnvironmentState,
    currentFPS: number
): IWebEnvironmentState => {
    // Validate input data structure
    if (!environmentData.pointCloud || !environmentData.meshData) {
        throw new Error('Invalid environment data structure');
    }

    // Optimize point cloud based on performance
    let optimizedPointCloud = environmentData.pointCloud;
    if (optimizedPointCloud.length > POINT_CLOUD_OPTIMIZATION_THRESHOLD) {
        const decimationFactor = Math.min(
            1.0,
            currentFPS < MIN_FPS ? 0.5 : currentFPS / TARGET_FPS
        );
        const stride = Math.max(1, Math.floor(1 / decimationFactor));
        optimizedPointCloud = new Float32Array(
            environmentData.pointCloud.filter((_, index) => index % stride === 0)
        );
    }

    // Apply mesh quality based on FPS
    let qualityLevel = MESH_QUALITY_LEVELS.HIGH;
    if (currentFPS < TARGET_FPS * 0.8) {
        qualityLevel = MESH_QUALITY_LEVELS.MEDIUM;
    } else if (currentFPS < TARGET_FPS * 0.6) {
        qualityLevel = MESH_QUALITY_LEVELS.LOW;
    }

    // Create optimized mesh data
    const optimizedMeshData = new ArrayBuffer(
        Math.ceil(environmentData.meshData.byteLength * qualityLevel)
    );
    new Uint8Array(optimizedMeshData).set(
        new Uint8Array(environmentData.meshData).subarray(
            0,
            optimizedMeshData.byteLength
        )
    );

    return {
        ...environmentData,
        pointCloud: optimizedPointCloud,
        meshData: optimizedMeshData,
        timestamp: Date.now()
    };
}, STATE_UPDATE_THROTTLE);