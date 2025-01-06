// External imports with versions for security tracking
import { describe, it, beforeEach, afterEach, expect, jest } from '@jest/globals'; // ^29.0.0
import { firstValueFrom } from 'rxjs'; // ^7.8.0
import { mockWebRTC } from '@testing-library/webrtc-mock'; // ^0.6.0
import { networkConditions } from '@testing-library/network-mock'; // ^2.1.0

// Internal imports
import { GameService } from '../../src/services/game.service';
import { ApiService } from '../../src/services/api.service';
import { server } from '../mocks/server';
import { handlers } from '../mocks/handlers';
import { 
    GameStates, 
    RenderQuality,
    GameEvents 
} from '../../src/interfaces/game.interface';

// Test constants
const mockGameId = 'test-game-123';
const mockSessionId = 'test-session-456';
const mockEnvironmentData = {
    scanQuality: 0.95,
    points: 1200000,
    range: 4.8
};
const mockFleetState = {
    devices: [],
    maxSize: 32,
    latency: 45
};
const mockPerformanceMetrics = {
    fps: 60,
    frameTime: 16.2,
    networkLatency: 48
};

describe('GameService Integration Tests', () => {
    let gameService: GameService;
    let apiService: ApiService;

    beforeEach(async () => {
        // Initialize MSW server
        server.listen();
        server.resetHandlers(...handlers);

        // Mock WebRTC connections
        mockWebRTC.setup();

        // Configure network conditions simulation
        networkConditions.set({
            latency: 45,
            jitter: 5,
            packetLoss: 0.01,
            bandwidth: 1000000
        });

        // Initialize services
        apiService = new ApiService();
        gameService = new GameService(apiService);
    });

    afterEach(() => {
        // Cleanup
        server.close();
        mockWebRTC.cleanup();
        networkConditions.reset();
        gameService.dispose();
    });

    describe('Game State Management', () => {
        it('should initialize game with correct state', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            
            expect(gameState).toEqual({
                gameId: mockGameId,
                sessionId: mockSessionId,
                state: GameStates.RUNNING,
                environmentData: null,
                renderState: {
                    resolution: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    quality: RenderQuality.HIGH,
                    lidarOverlayEnabled: true
                },
                fps: 60
            });
        });

        it('should maintain CRDT state consistency', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            const crdtState = await firstValueFrom(gameService['crdtState$']);
            
            expect(crdtState.gameId).toBe(mockGameId);
            expect(crdtState.sessionId).toBe(mockSessionId);
            expect(crdtState.state).toBe(GameStates.RUNNING);
        });

        it('should handle environment updates correctly', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            await gameService['handleEnvironmentUpdate'](mockEnvironmentData);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            expect(gameState.environmentData).toEqual(mockEnvironmentData);
        });
    });

    describe('Performance Requirements', () => {
        it('should maintain minimum 60 FPS under load', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Simulate heavy load
            for (let i = 0; i < 1000; i++) {
                await gameService['handleEnvironmentUpdate'](mockEnvironmentData);
            }
            
            const metrics = await firstValueFrom(gameService['performanceMetrics$']);
            expect(metrics.fps).toBeGreaterThanOrEqual(60);
        });

        it('should keep network latency under 50ms', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            await gameService['syncWithFleet'](mockEnvironmentData);
            
            const metrics = await firstValueFrom(gameService['performanceMetrics$']);
            expect(metrics.networkLatency).toBeLessThanOrEqual(50);
        });

        it('should handle 32 device fleet efficiently', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Simulate 32 device fleet
            const fleetState = { ...mockFleetState, devices: Array(32).fill({}) };
            await gameService['syncWithFleet'](mockEnvironmentData);
            
            const metrics = await firstValueFrom(gameService['performanceMetrics$']);
            expect(metrics.networkLatency).toBeLessThanOrEqual(50);
            expect(metrics.fps).toBeGreaterThanOrEqual(60);
        });
    });

    describe('Error Handling', () => {
        it('should handle network disconnections gracefully', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Simulate network failure
            networkConditions.set({ packetLoss: 1 });
            await gameService['syncWithFleet'](mockEnvironmentData);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            expect(gameState.state).not.toBe(GameStates.ERROR);
        });

        it('should recover from state synchronization failures', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Corrupt CRDT state
            gameService['crdtDoc'] = null as any;
            await gameService['syncWithFleet'](mockEnvironmentData);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            expect(gameState.state).toBe(GameStates.RUNNING);
        });

        it('should handle WebRTC connection issues', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Simulate WebRTC failure
            mockWebRTC.disconnect();
            await gameService['syncWithFleet'](mockEnvironmentData);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            expect(gameState.state).not.toBe(GameStates.ERROR);
        });

        it('should handle performance degradation appropriately', async () => {
            await gameService.startGame(mockGameId, mockSessionId);
            
            // Simulate performance issues
            const lowPerformanceMetrics = { ...mockPerformanceMetrics, fps: 30 };
            gameService['performanceMetrics$'].next(lowPerformanceMetrics);
            
            const gameState = await firstValueFrom(gameService['gameState$']);
            expect(gameState.renderState.quality).not.toBe(RenderQuality.HIGH);
        });
    });
});