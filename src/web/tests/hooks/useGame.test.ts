// External imports with versions for security tracking
import { renderHook, act } from '@testing-library/react-hooks'; // ^8.0.1
import { jest, describe, beforeEach, it, expect } from '@jest/globals'; // ^29.5.0
import { waitFor } from '@testing-library/react'; // ^14.0.0

// Internal imports
import { useGame } from '../../src/hooks/useGame';
import { GameService } from '../../src/services/game.service';
import {
    IWebGameState,
    GameStates,
    IWebEnvironmentState,
    IPerformanceMetrics,
    RenderQuality
} from '../../src/interfaces/game.interface';

// Mock GameService
jest.mock('../../src/services/game.service');

describe('useGame hook', () => {
    // Test constants from technical specifications
    const TARGET_FPS = 60;
    const MAX_LATENCY = 50;
    const MEMORY_THRESHOLD = 0.8;
    const POINT_CLOUD_SIZE = 1_200_000;
    const SESSION_ID = 'test-session-123';

    // Mock environment data
    const mockEnvironmentData: IWebEnvironmentState = {
        meshData: new ArrayBuffer(1000),
        pointCloud: new Float32Array(POINT_CLOUD_SIZE),
        classifiedObjects: [],
        timestamp: Date.now()
    };

    beforeEach(() => {
        // Reset mocks
        jest.clearAllMocks();
        
        // Mock performance API
        global.performance = {
            now: jest.fn(() => Date.now()),
            memory: {
                usedJSHeapSize: 100,
                jsHeapSizeLimit: 1000
            }
        } as any;

        // Mock requestAnimationFrame
        global.requestAnimationFrame = jest.fn(cb => setTimeout(cb, 16));
        global.cancelAnimationFrame = jest.fn(id => clearTimeout(id));

        // Mock GameService implementation
        (GameService as jest.Mock).mockImplementation(() => ({
            startGameSession: jest.fn().mockResolvedValue(undefined),
            updateEnvironment: jest.fn().mockReturnValue(mockEnvironmentData),
            processLiDARData: jest.fn().mockResolvedValue(undefined),
            getPerformanceMetrics: jest.fn().mockReturnValue({
                fps: TARGET_FPS,
                frameTime: 16,
                memoryUsage: 0.5
            })
        }));
    });

    it('should maintain performance targets', async () => {
        const { result } = renderHook(() => useGame(SESSION_ID));

        await act(async () => {
            await result.current.startGame();
        });

        // Verify initial performance metrics
        expect(result.current.performanceMetrics.fps).toBeGreaterThanOrEqual(TARGET_FPS);
        expect(result.current.performanceMetrics.frameTime).toBeLessThanOrEqual(16.67);
        expect(result.current.performanceMetrics.memoryUsage).toBeLessThan(MEMORY_THRESHOLD);

        // Simulate environment updates
        await act(async () => {
            result.current.updateEnvironment(mockEnvironmentData);
        });

        // Verify performance after updates
        expect(result.current.performanceMetrics.fps).toBeGreaterThanOrEqual(TARGET_FPS * 0.9);
        expect(result.current.gameState.state).toBe(GameStates.RUNNING);
    });

    it('should handle LiDAR updates efficiently', async () => {
        const { result } = renderHook(() => useGame(SESSION_ID, {
            lidarEnabled: true,
            adaptiveQuality: true
        }));

        await act(async () => {
            await result.current.startGame();
        });

        // Simulate high-load LiDAR updates
        const highLoadData = {
            ...mockEnvironmentData,
            pointCloud: new Float32Array(POINT_CLOUD_SIZE * 2)
        };

        await act(async () => {
            result.current.updateEnvironment(highLoadData);
        });

        // Verify adaptive quality management
        await waitFor(() => {
            expect(result.current.gameState.renderState.quality).toBe(RenderQuality.MEDIUM);
            expect(result.current.performanceMetrics.fps).toBeGreaterThanOrEqual(TARGET_FPS * 0.8);
        });
    });

    it('should optimize memory usage under load', async () => {
        const { result } = renderHook(() => useGame(SESSION_ID));

        await act(async () => {
            await result.current.startGame();
        });

        // Simulate memory pressure
        (global.performance as any).memory.usedJSHeapSize = 900;
        (global.performance as any).memory.jsHeapSizeLimit = 1000;

        await act(async () => {
            result.current.updateEnvironment(mockEnvironmentData);
        });

        // Verify memory optimization
        await waitFor(() => {
            expect(result.current.performanceMetrics.memoryUsage).toBeLessThan(MEMORY_THRESHOLD);
            expect(result.current.gameState.renderState.quality).not.toBe(RenderQuality.HIGH);
        });
    });

    it('should cleanup resources properly', async () => {
        const { result, unmount } = renderHook(() => useGame(SESSION_ID));

        await act(async () => {
            await result.current.startGame();
        });

        // Track animation frame and interval cleanup
        const animationFrameSpy = jest.spyOn(window, 'cancelAnimationFrame');
        const clearIntervalSpy = jest.spyOn(window, 'clearInterval');

        // Unmount hook
        unmount();

        // Verify cleanup
        expect(animationFrameSpy).toHaveBeenCalled();
        expect(clearIntervalSpy).toHaveBeenCalled();
        expect(GameService.prototype.dispose).toHaveBeenCalled();
    });

    it('should recover from performance degradation', async () => {
        const { result } = renderHook(() => useGame(SESSION_ID, {
            adaptiveQuality: true
        }));

        await act(async () => {
            await result.current.startGame();
        });

        // Simulate performance drop
        (GameService as jest.Mock).mockImplementation(() => ({
            ...GameService.prototype,
            getPerformanceMetrics: jest.fn().mockReturnValue({
                fps: TARGET_FPS * 0.5,
                frameTime: 32,
                memoryUsage: 0.7
            })
        }));

        await act(async () => {
            result.current.updateEnvironment(mockEnvironmentData);
        });

        // Verify recovery measures
        await waitFor(() => {
            expect(result.current.gameState.renderState.quality).toBe(RenderQuality.LOW);
            expect(result.current.performanceMetrics.fps).toBeGreaterThanOrEqual(TARGET_FPS * 0.6);
        });
    });

    it('should maintain state consistency during rapid updates', async () => {
        const { result } = renderHook(() => useGame(SESSION_ID));

        await act(async () => {
            await result.current.startGame();
        });

        // Simulate rapid environment updates
        const updates = Array(10).fill(mockEnvironmentData);
        
        await act(async () => {
            await Promise.all(updates.map(update => 
                result.current.updateEnvironment(update)
            ));
        });

        // Verify state consistency
        expect(result.current.gameState.state).toBe(GameStates.RUNNING);
        expect(result.current.performanceMetrics.fps).toBeGreaterThanOrEqual(TARGET_FPS * 0.8);
        expect(result.current.gameState.environmentData).toBeDefined();
    });
});