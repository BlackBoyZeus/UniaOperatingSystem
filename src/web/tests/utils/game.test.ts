import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import {
    calculateFPS,
    validateGameState,
    processEnvironmentData,
    optimizeRenderConfig
} from '../../src/utils/game.utils';
import {
    GameState,
    RenderQuality,
    EnvironmentQuality,
    PointCloudDensity
} from '../../src/types/game.types';
import {
    IWebGameState,
    IWebEnvironmentState,
    IWebRenderState,
    IPointCloudData,
    IMeshData
} from '../../src/interfaces/game.interface';

// Mock game state for testing
const MOCK_GAME_STATE: IWebGameState = {
    gameId: 'test_game_123',
    sessionId: 'test_session_456',
    state: GameState.RUNNING,
    environmentData: {
        pointCloud: new Float32Array(1000),
        meshData: new ArrayBuffer(1000),
        classifiedObjects: [
            {
                id: 'obj_1',
                type: 'static',
                position: { x: 0, y: 0, z: 0 }
            }
        ],
        timestamp: Date.now()
    },
    renderState: {
        resolution: { width: 1920, height: 1080 },
        quality: RenderQuality.HIGH,
        lidarOverlayEnabled: true
    },
    fps: 60 as ValidFPS
};

describe('calculateFPS', () => {
    const TARGET_FPS = 60;
    const MIN_FPS = 30;
    let mockTime: number;

    beforeEach(() => {
        mockTime = Date.now();
        jest.useFakeTimers();
    });

    afterEach(() => {
        jest.useRealTimers();
    });

    it('should calculate correct FPS with ideal frame times', () => {
        const idealFrameTime = 1000 / TARGET_FPS;
        const result = calculateFPS(
            mockTime + idealFrameTime,
            mockTime,
            MOCK_GAME_STATE.renderState
        );

        expect(result.fps).toBe(TARGET_FPS);
        expect(result.shouldOptimize).toBe(false);
    });

    it('should detect performance issues when FPS drops below threshold', () => {
        const lowFrameTime = 1000 / (TARGET_FPS * 0.8); // 20% slower
        const result = calculateFPS(
            mockTime + lowFrameTime,
            mockTime,
            MOCK_GAME_STATE.renderState
        );

        expect(result.fps).toBeLessThan(TARGET_FPS);
        expect(result.shouldOptimize).toBe(true);
    });

    it('should clamp FPS to minimum value', () => {
        const veryLowFrameTime = 1000 / (MIN_FPS * 0.5); // 50% of min FPS
        const result = calculateFPS(
            mockTime + veryLowFrameTime,
            mockTime,
            MOCK_GAME_STATE.renderState
        );

        expect(result.fps).toBe(MIN_FPS);
        expect(result.shouldOptimize).toBe(true);
    });

    it('should handle frame time spikes gracefully', () => {
        const spikeFrameTime = 100; // Sudden 100ms frame
        const result = calculateFPS(
            mockTime + spikeFrameTime,
            mockTime,
            MOCK_GAME_STATE.renderState
        );

        expect(result.fps).toBeGreaterThanOrEqual(MIN_FPS);
        expect(result.shouldOptimize).toBe(true);
    });
});

describe('validateGameState', () => {
    it('should validate complete and valid game state', () => {
        const result = validateGameState(MOCK_GAME_STATE);
        expect(result.isValid).toBe(true);
        expect(result.conflicts).toHaveLength(0);
    });

    it('should detect missing required fields', () => {
        const invalidState = { ...MOCK_GAME_STATE, gameId: undefined };
        const result = validateGameState(invalidState);
        
        expect(result.isValid).toBe(false);
        expect(result.conflicts).toContain('Invalid game or session ID');
    });

    it('should validate environment data freshness', () => {
        const staleState = {
            ...MOCK_GAME_STATE,
            environmentData: {
                ...MOCK_GAME_STATE.environmentData!,
                timestamp: Date.now() - 1000 // 1 second old
            }
        };
        const result = validateGameState(staleState);

        expect(result.isValid).toBe(false);
        expect(result.conflicts).toContain('Environment data outdated');
    });

    it('should validate render quality settings', () => {
        const invalidQualityState = {
            ...MOCK_GAME_STATE,
            renderState: {
                ...MOCK_GAME_STATE.renderState,
                quality: 'INVALID' as RenderQuality
            }
        };
        const result = validateGameState(invalidQualityState);

        expect(result.isValid).toBe(false);
        expect(result.conflicts).toContain('Invalid render quality setting');
    });
});

describe('processEnvironmentData', () => {
    const mockEnvironmentData: IWebEnvironmentState = {
        pointCloud: new Float32Array(1_000_000), // 1M points
        meshData: new ArrayBuffer(2_000_000), // 2MB mesh
        classifiedObjects: [
            {
                id: 'test_obj_1',
                type: 'static',
                position: { x: 0, y: 0, z: 0 }
            }
        ],
        timestamp: Date.now()
    };

    it('should optimize point cloud when exceeding threshold', () => {
        const result = processEnvironmentData(mockEnvironmentData, 45); // Below target FPS
        expect(result.pointCloud.length).toBeLessThan(mockEnvironmentData.pointCloud.length);
    });

    it('should maintain point cloud quality at high FPS', () => {
        const result = processEnvironmentData(mockEnvironmentData, TARGET_FPS);
        expect(result.pointCloud.length).toBe(mockEnvironmentData.pointCloud.length);
    });

    it('should adjust mesh quality based on FPS', () => {
        const lowFPSResult = processEnvironmentData(mockEnvironmentData, 30);
        const highFPSResult = processEnvironmentData(mockEnvironmentData, 60);

        expect(lowFPSResult.meshData.byteLength).toBeLessThan(highFPSResult.meshData.byteLength);
    });

    it('should throw error for invalid environment data', () => {
        const invalidData = { timestamp: Date.now() } as IWebEnvironmentState;
        expect(() => processEnvironmentData(invalidData, 60)).toThrow();
    });
});

describe('optimizeRenderConfig', () => {
    const mockRenderState: IWebRenderState = {
        resolution: { width: 1920, height: 1080 },
        quality: RenderQuality.HIGH,
        lidarOverlayEnabled: true
    };

    it('should maintain high quality at target FPS', () => {
        const result = optimizeRenderConfig(mockRenderState, TARGET_FPS);
        expect(result.quality).toBe(RenderQuality.HIGH);
        expect(result.resolution).toEqual(mockRenderState.resolution);
    });

    it('should reduce quality when FPS drops', () => {
        const result = optimizeRenderConfig(mockRenderState, TARGET_FPS * 0.7);
        expect(result.quality).toBe(RenderQuality.MEDIUM);
    });

    it('should disable LiDAR overlay under heavy load', () => {
        const result = optimizeRenderConfig(mockRenderState, MIN_FPS);
        expect(result.lidarOverlayEnabled).toBe(false);
    });

    it('should scale resolution based on performance', () => {
        const result = optimizeRenderConfig(mockRenderState, TARGET_FPS * 0.5);
        expect(result.resolution.width).toBeLessThan(mockRenderState.resolution.width);
        expect(result.resolution.height).toBeLessThan(mockRenderState.resolution.height);
    });
});