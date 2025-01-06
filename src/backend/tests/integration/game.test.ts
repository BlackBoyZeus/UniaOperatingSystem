import { Container } from 'inversify';
import { performance } from 'performance-now'; // version: 2.1.0
import '@testing-library/jest-dom'; // version: 5.16.5

import { GameService } from '../../src/services/game/GameService';
import { GameState } from '../../src/core/game/GameState';
import { 
    IGameState, 
    IEnvironmentState, 
    IPhysicsState,
    Vector3,
    IPerformanceMetrics 
} from '../../interfaces/game.interface';

// Constants for test configuration
const TEST_TIMEOUT = 10000; // 10 second timeout for async tests
const PERFORMANCE_REQUIREMENTS = {
    MAX_LATENCY: 50, // 50ms max latency
    MIN_FPS: 60,
    MAX_FLEET_SIZE: 32,
    SCAN_RATE: 30 // 30Hz scan rate
};

// Test data generators
const createMockEnvironmentState = (): IEnvironmentState => ({
    timestamp: Date.now(),
    scanQuality: 1.0,
    pointCount: 100000,
    classifiedObjects: [],
    lidarMetrics: {
        scanRate: PERFORMANCE_REQUIREMENTS.SCAN_RATE,
        resolution: 0.01,
        effectiveRange: 5.0,
        pointDensity: 1000
    }
});

const createMockPhysicsState = (): IPhysicsState => ({
    timestamp: Date.now(),
    objects: [],
    collisions: [],
    simulationLatency: 0
});

describe('GameService Integration Tests', () => {
    let container: Container;
    let gameService: GameService;
    let testSessionId: string;

    // Setup test container and dependencies
    beforeAll(() => {
        container = new Container();
        container.bind<GameService>(GameService).toSelf();
        container.bind<GameState>(GameState).toSelf();

        // Configure performance monitoring
        container.bind('PerformanceMonitor').toConstantValue({
            getMetrics: () => ({
                temperature: 60,
                batteryLevel: 80,
                memoryUsage: 0.5
            }),
            recordMetrics: jest.fn()
        });

        gameService = container.get<GameService>(GameService);
    });

    // Clean up after each test
    afterEach(async () => {
        if (testSessionId) {
            await gameService.endSession(testSessionId);
            testSessionId = '';
        }
    });

    describe('Game Session Management', () => {
        it('should create and initialize game session within latency requirements', async () => {
            const startTime = performance();
            
            // Create new game session
            const sessionConfig = {
                fleetSize: 1,
                powerMode: 'BALANCED',
                thermalPolicy: {
                    warningThreshold: 75,
                    criticalThreshold: 85
                }
            };

            const result = await gameService.createSession(
                'test-session',
                'test-fleet',
                sessionConfig
            );

            testSessionId = 'test-session';

            // Validate session creation latency
            const latency = performance() - startTime;
            expect(latency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);

            // Validate session state
            expect(result).toBeDefined();
            expect(result.gameId).toBe('test-session');
            expect(result.fleetId).toBe('test-fleet');
        });

        it('should handle concurrent session operations with CRDT consistency', async () => {
            testSessionId = 'concurrent-test';
            const sessionConfig = {
                fleetSize: 2,
                powerMode: 'BALANCED',
                thermalPolicy: {
                    warningThreshold: 75,
                    criticalThreshold: 85
                }
            };

            // Create initial session
            await gameService.createSession(
                testSessionId,
                'concurrent-fleet',
                sessionConfig
            );

            // Simulate concurrent state updates
            const updates = Array(5).fill(null).map(async (_, index) => {
                const envState = createMockEnvironmentState();
                envState.pointCount += index * 1000;
                return gameService.processEnvironmentUpdate(testSessionId, envState);
            });

            await Promise.all(updates);

            // Verify state consistency
            const finalState = await gameService.getSessionState(testSessionId);
            expect(finalState.environment.pointCount).toBeGreaterThan(100000);
            expect(finalState.timestamp).toBeDefined();
        });
    });

    describe('Environment Synchronization', () => {
        beforeEach(async () => {
            testSessionId = 'env-sync-test';
            await gameService.createSession(
                testSessionId,
                'env-fleet',
                {
                    fleetSize: 1,
                    powerMode: 'BALANCED',
                    thermalPolicy: {
                        warningThreshold: 75,
                        criticalThreshold: 85
                    }
                }
            );
        });

        it('should process environment updates within latency requirements', async () => {
            const envState = createMockEnvironmentState();
            const startTime = performance();

            await gameService.processEnvironmentUpdate(testSessionId, envState);

            const latency = performance() - startTime;
            expect(latency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);

            const state = await gameService.getSessionState(testSessionId);
            expect(state.environment.scanQuality).toBe(envState.scanQuality);
            expect(state.environment.pointCount).toBe(envState.pointCount);
        });

        it('should maintain LiDAR processing performance at 30Hz', async () => {
            const iterations = 30; // Test 1 second of updates at 30Hz
            const results: number[] = [];

            for (let i = 0; i < iterations; i++) {
                const startTime = performance();
                await gameService.processEnvironmentUpdate(
                    testSessionId,
                    createMockEnvironmentState()
                );
                results.push(performance() - startTime);
            }

            // Validate processing latency
            const avgLatency = results.reduce((a, b) => a + b) / results.length;
            expect(avgLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);

            // Verify scan rate maintenance
            const scanRate = 1000 / avgLatency;
            expect(scanRate).toBeGreaterThanOrEqual(PERFORMANCE_REQUIREMENTS.SCAN_RATE);
        });
    });

    describe('Fleet Coordination', () => {
        it('should scale to maximum fleet size while maintaining performance', async () => {
            testSessionId = 'fleet-scale-test';
            
            // Create session with max fleet size
            await gameService.createSession(
                testSessionId,
                'scale-fleet',
                {
                    fleetSize: PERFORMANCE_REQUIREMENTS.MAX_FLEET_SIZE,
                    powerMode: 'BALANCED',
                    thermalPolicy: {
                        warningThreshold: 75,
                        criticalThreshold: 85
                    }
                }
            );

            // Simulate fleet-wide state updates
            const fleetUpdates = Array(PERFORMANCE_REQUIREMENTS.MAX_FLEET_SIZE)
                .fill(null)
                .map(async (_, index) => {
                    const startTime = performance();
                    await gameService.synchronizeState(testSessionId, {
                        gameId: testSessionId,
                        sessionId: `device-${index}`,
                        fleetId: 'scale-fleet',
                        deviceCount: PERFORMANCE_REQUIREMENTS.MAX_FLEET_SIZE,
                        timestamp: Date.now(),
                        environment: createMockEnvironmentState(),
                        physics: createMockPhysicsState(),
                        metrics: {
                            stateUpdateLatency: 0,
                            lidarProcessingLatency: 0,
                            physicsSimulationLatency: 0,
                            fleetSyncLatency: 0
                        }
                    });
                    return performance() - startTime;
                });

            const latencies = await Promise.all(fleetUpdates);
            const maxLatency = Math.max(...latencies);
            const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;

            expect(maxLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);
            expect(avgLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY * 0.8);
        });

        it('should maintain CRDT consistency across fleet updates', async () => {
            testSessionId = 'fleet-consistency-test';
            
            await gameService.createSession(
                testSessionId,
                'consistency-fleet',
                {
                    fleetSize: 3,
                    powerMode: 'BALANCED',
                    thermalPolicy: {
                        warningThreshold: 75,
                        criticalThreshold: 85
                    }
                }
            );

            // Simulate concurrent updates from multiple fleet members
            const updates = Array(3).fill(null).map(async (_, index) => {
                const envState = createMockEnvironmentState();
                envState.pointCount = (index + 1) * 100000;
                
                await gameService.processEnvironmentUpdate(testSessionId, envState);
                await gameService.synchronizeState(testSessionId, {
                    gameId: testSessionId,
                    sessionId: `device-${index}`,
                    fleetId: 'consistency-fleet',
                    deviceCount: 3,
                    timestamp: Date.now(),
                    environment: envState,
                    physics: createMockPhysicsState(),
                    metrics: {
                        stateUpdateLatency: 0,
                        lidarProcessingLatency: 0,
                        physicsSimulationLatency: 0,
                        fleetSyncLatency: 0
                    }
                });
            });

            await Promise.all(updates);

            // Verify state consistency
            const finalState = await gameService.getSessionState(testSessionId);
            expect(finalState.environment.pointCount).toBe(300000);
            expect(finalState.deviceCount).toBe(3);
        });
    });

    describe('Performance Monitoring', () => {
        beforeEach(async () => {
            testSessionId = 'perf-test';
            await gameService.createSession(
                testSessionId,
                'perf-fleet',
                {
                    fleetSize: 1,
                    powerMode: 'BALANCED',
                    thermalPolicy: {
                        warningThreshold: 75,
                        criticalThreshold: 85
                    }
                }
            );
        });

        it('should maintain performance metrics within requirements', async () => {
            const iterations = 100;
            const metrics: IPerformanceMetrics[] = [];

            for (let i = 0; i < iterations; i++) {
                await gameService.processEnvironmentUpdate(
                    testSessionId,
                    createMockEnvironmentState()
                );
                
                const state = await gameService.getSessionState(testSessionId);
                metrics.push(state.metrics);
            }

            // Validate performance metrics
            metrics.forEach(metric => {
                expect(metric.stateUpdateLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);
                expect(metric.lidarProcessingLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);
                expect(metric.fleetSyncLatency).toBeLessThanOrEqual(PERFORMANCE_REQUIREMENTS.MAX_LATENCY);
            });
        });
    });
});