import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { SessionService } from '../../src/services/session/SessionService';
import { 
    ISession, 
    ISessionConfig, 
    ISessionState, 
    SessionStatus, 
    NetworkMetrics 
} from '../../src/interfaces/session.interface';
import {
    setupTestFleet,
    setupTestGameState,
    cleanupTestData,
    mockNetworkConditions
} from '../utils/testHelpers';

describe('SessionService', () => {
    let sessionService: SessionService;
    let mockFleetService: jest.Mocked<any>;
    let mockWebRTCService: jest.Mocked<any>;
    let defaultConfig: ISessionConfig;

    beforeEach(async () => {
        // Initialize mock services
        mockFleetService = {
            createFleet: jest.fn(),
            validateFleetSize: jest.fn(),
            updateFleetState: jest.fn()
        };

        mockWebRTCService = {
            createPeerConnection: jest.fn(),
            getConnectionStats: jest.fn(),
            monitorConnection: jest.fn()
        };

        // Setup default session configuration
        defaultConfig = {
            maxParticipants: 32,
            networkConfig: {
                meshTopology: 'full',
                maxLatency: 50,
                autoRecoveryEnabled: true
            },
            scanRate: 30,
            performanceThresholds: {
                maxLatency: 50,
                maxPacketLoss: 0.01,
                minSyncRate: 20,
                minBatteryLevel: 20,
                minFrameRate: 60,
                minScanQuality: 0.95,
                maxCpuUsage: 80,
                maxMemoryUsage: 85
            },
            autoRecoveryEnabled: true,
            meshTopology: 'full',
            stateValidation: true,
            compressionEnabled: true,
            encryptionEnabled: true
        };

        // Initialize session service with mocks
        sessionService = new SessionService(mockFleetService, mockWebRTCService);
    });

    afterEach(async () => {
        await cleanupTestData();
        jest.clearAllMocks();
    });

    describe('Session Creation', () => {
        it('should create a new session with valid configuration', async () => {
            const { fleet } = await setupTestFleet(32);
            mockFleetService.createFleet.mockResolvedValue(fleet);

            const session = await sessionService.createSession(defaultConfig);

            expect(session).toBeDefined();
            expect(session.state.status).toBe(SessionStatus.INITIALIZING);
            expect(session.config.maxParticipants).toBe(32);
            expect(mockFleetService.createFleet).toHaveBeenCalledWith(
                expect.objectContaining({ maxDevices: 32 })
            );
        });

        it('should reject session creation when fleet size exceeds limit', async () => {
            const invalidConfig = { ...defaultConfig, maxParticipants: 33 };

            await expect(sessionService.createSession(invalidConfig))
                .rejects.toThrow('Maximum participants cannot exceed 32');
        });

        it('should initialize session with correct performance monitoring', async () => {
            const { fleet } = await setupTestFleet(32);
            mockFleetService.createFleet.mockResolvedValue(fleet);

            const session = await sessionService.createSession(defaultConfig);

            expect(session.state.performanceMetrics).toBeDefined();
            expect(session.state.performanceMetrics.averageLatency).toBe(0);
            expect(session.state.performanceMetrics.syncRate).toBe(0);
        });
    });

    describe('Session Participant Management', () => {
        let activeSession: ISession;

        beforeEach(async () => {
            const { fleet } = await setupTestFleet(1);
            mockFleetService.createFleet.mockResolvedValue(fleet);
            activeSession = await sessionService.createSession(defaultConfig);
        });

        it('should add participant within fleet size limit', async () => {
            mockFleetService.validateFleetSize.mockResolvedValue(true);
            const participant = {
                id: 'test-participant',
                capabilities: { lidarSupport: true, networkBandwidth: 1500 }
            };

            await sessionService.joinSession(activeSession.sessionId, participant);

            expect(activeSession.state.activeParticipants).toBe(1);
            expect(mockFleetService.updateFleetState).toHaveBeenCalled();
        });

        it('should reject participant when fleet is full', async () => {
            mockFleetService.validateFleetSize.mockResolvedValue(false);
            const participant = {
                id: 'test-participant',
                capabilities: { lidarSupport: true, networkBandwidth: 1500 }
            };

            await expect(sessionService.joinSession(activeSession.sessionId, participant))
                .rejects.toThrow('Fleet has reached maximum capacity');
        });

        it('should validate participant capabilities', async () => {
            const invalidParticipant = {
                id: 'test-participant',
                capabilities: { lidarSupport: false, networkBandwidth: 500 }
            };

            await expect(sessionService.joinSession(activeSession.sessionId, invalidParticipant))
                .rejects.toThrow('Insufficient capabilities');
        });
    });

    describe('Session State Management', () => {
        let activeSession: ISession;

        beforeEach(async () => {
            const { fleet } = await setupTestFleet(32);
            mockFleetService.createFleet.mockResolvedValue(fleet);
            activeSession = await sessionService.createSession(defaultConfig);
        });

        it('should synchronize state within latency requirements', async () => {
            const gameState = await setupTestGameState();
            const startTime = Date.now();

            await sessionService.updateSessionState(activeSession.sessionId, gameState.gameState);

            const syncLatency = Date.now() - startTime;
            expect(syncLatency).toBeLessThanOrEqual(50);
            expect(activeSession.state.lastUpdate).toBeGreaterThan(startTime);
        });

        it('should handle state conflicts with CRDT', async () => {
            const conflictingState = await setupTestGameState();
            const originalState = activeSession.gameState;

            await sessionService.handleStateConflict(
                activeSession.sessionId,
                conflictingState.gameState
            );

            expect(activeSession.gameState).not.toEqual(originalState);
            expect(activeSession.state.errorCount).toBe(0);
        });

        it('should validate state integrity', async () => {
            const { gameState } = await setupTestGameState();
            const isValid = sessionService.validateSessionState(activeSession.sessionId);

            expect(isValid).toBe(true);
            expect(activeSession.state.warningCount).toBe(0);
        });
    });

    describe('Session Performance Monitoring', () => {
        let activeSession: ISession;

        beforeEach(async () => {
            const { fleet } = await setupTestFleet(32);
            mockFleetService.createFleet.mockResolvedValue(fleet);
            activeSession = await sessionService.createSession(defaultConfig);
        });

        it('should monitor network latency within threshold', async () => {
            mockWebRTCService.getConnectionStats.mockResolvedValue({ latency: 45 });

            await sessionService.monitorSessionHealth(activeSession.sessionId);

            expect(activeSession.state.averageLatency).toBeLessThanOrEqual(50);
            expect(activeSession.state.status).toBe(SessionStatus.ACTIVE);
        });

        it('should detect and handle performance degradation', async () => {
            mockWebRTCService.getConnectionStats.mockResolvedValue({ latency: 75 });

            await sessionService.monitorSessionHealth(activeSession.sessionId);

            expect(activeSession.state.status).toBe(SessionStatus.DEGRADED);
            expect(activeSession.state.recoveryAttempts).toBeGreaterThan(0);
        });

        it('should track comprehensive performance metrics', async () => {
            const metrics = await sessionService.getSessionState(activeSession.sessionId);

            expect(metrics.performanceMetrics).toEqual(
                expect.objectContaining({
                    averageLatency: expect.any(Number),
                    packetLoss: expect.any(Number),
                    syncRate: expect.any(Number),
                    cpuUsage: expect.any(Number),
                    memoryUsage: expect.any(Number),
                    batteryLevel: expect.any(Number),
                    networkBandwidth: expect.any(Number),
                    scanQuality: expect.any(Number),
                    frameRate: expect.any(Number)
                })
            );
        });
    });

    describe('Session Error Handling', () => {
        let activeSession: ISession;

        beforeEach(async () => {
            const { fleet } = await setupTestFleet(32);
            mockFleetService.createFleet.mockResolvedValue(fleet);
            activeSession = await sessionService.createSession(defaultConfig);
        });

        it('should handle network failures with recovery', async () => {
            mockWebRTCService.getConnectionStats.mockRejectedValue(new Error('Network failure'));

            await sessionService.monitorSessionHealth(activeSession.sessionId);

            expect(activeSession.recoveryMode).toBe(true);
            expect(activeSession.errors.length).toBeGreaterThan(0);
        });

        it('should implement automatic recovery within retry limits', async () => {
            for (let i = 0; i < 3; i++) {
                await sessionService.attemptSessionRecovery(activeSession.sessionId);
            }

            expect(activeSession.recoveryMode).toBe(false);
            expect(activeSession.state.recoveryAttempts).toBeLessThanOrEqual(3);
        });

        it('should generate diagnostic data for troubleshooting', async () => {
            const diagnostics = await sessionService.generateSessionDiagnostics(activeSession.sessionId);

            expect(diagnostics).toEqual(
                expect.objectContaining({
                    sessionId: activeSession.sessionId,
                    status: expect.any(String),
                    errors: expect.any(Array),
                    metrics: expect.any(Object),
                    timestamp: expect.any(Number)
                })
            );
        });
    });
});