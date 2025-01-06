import { describe, beforeEach, afterEach, it, expect, jest } from '@jest/globals';
import { retry } from 'jest-retry';

import { FleetManager } from '../../src/core/fleet/FleetManager';
import { 
    IFleet, 
    IFleetMember, 
    IFleetState, 
    IMeshTopology,
    FleetRole,
    FleetStatus,
    MeshTopologyType 
} from '../../src/interfaces/fleet.interface';
import { 
    setupTestFleet, 
    cleanupTestData, 
    setupPerformanceMonitoring 
} from '../utils/testHelpers';

// Test configuration constants
const TEST_TIMEOUT = 10000;
const MAX_TEST_FLEET_SIZE = 32;
const SYNC_TEST_INTERVAL = 50;
const LATENCY_THRESHOLD = 50;
const RETRY_ATTEMPTS = 3;

describe('Fleet Management System Integration Tests', () => {
    let fleetManager: FleetManager;
    let testFleet: IFleet;
    let performanceMetrics: { latency: number; meshQuality: number; convergenceTime: number };

    beforeEach(async () => {
        // Setup test environment with monitoring
        const testSetup = await setupTestFleet(MAX_TEST_FLEET_SIZE, {
            maxDevices: MAX_TEST_FLEET_SIZE,
            meshConfig: {
                topology: MeshTopologyType.HYBRID,
                maxPeers: MAX_TEST_FLEET_SIZE,
                reconnectStrategy: {
                    maxAttempts: RETRY_ATTEMPTS,
                    backoffMultiplier: 1.5,
                    initialDelay: 100,
                    maxDelay: 1000
                }
            }
        });

        testFleet = testSetup.fleet;
        fleetManager = new FleetManager(testFleet, {
            encryptionEnabled: true,
            authenticationMethod: 'certificate',
            accessControl: {
                allowedDevices: [],
                bannedDevices: [],
                joinPolicy: 'open',
                rolePermissions: new Map()
            }
        });

        performanceMetrics = {
            latency: 0,
            meshQuality: 1.0,
            convergenceTime: 0
        };
    });

    afterEach(async () => {
        await cleanupTestData({
            fleetId: testFleet.id,
            verifyCleanup: true
        });
        fleetManager.dispose();
    });

    describe('Fleet Creation and Scaling', () => {
        it('should create and initialize fleet with maximum capacity', async () => {
            const startTime = Date.now();
            const fleet = await fleetManager.createFleet(testFleet);

            expect(fleet.id).toBeDefined();
            expect(fleet.members.length).toBeLessThanOrEqual(MAX_TEST_FLEET_SIZE);
            expect(fleet.meshConfig.topology).toBe(MeshTopologyType.HYBRID);
            expect(Date.now() - startTime).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        }, TEST_TIMEOUT);

        it('should enforce fleet size limits and handle overflow gracefully', async () => {
            const oversizedFleet = { ...testFleet, maxDevices: MAX_TEST_FLEET_SIZE + 1 };
            await expect(fleetManager.createFleet(oversizedFleet))
                .rejects
                .toThrow(`Maximum fleet size is ${MAX_TEST_FLEET_SIZE}`);
        });

        it('should validate member capabilities during fleet formation', async () => {
            const invalidMember: IFleetMember = {
                ...testFleet.members[0],
                capabilities: {
                    ...testFleet.members[0].capabilities,
                    lidarSupport: false
                }
            };

            await expect(fleetManager.joinFleet(invalidMember.id, invalidMember.capabilities))
                .rejects
                .toThrow('LiDAR support required for fleet membership');
        });
    });

    describe('State Synchronization and Mesh Networking', () => {
        it('should maintain state consistency across fleet members', async () => {
            await retry(async () => {
                const startTime = Date.now();
                const fleet = await fleetManager.createFleet(testFleet);
                
                // Simulate state changes across members
                for (const member of fleet.members) {
                    await fleetManager.joinFleet(member.id, member.capabilities);
                }

                const fleetState = fleetManager.getFleetState();
                const syncTime = Date.now() - startTime;

                expect(fleetState.members.length).toBe(fleet.members.length);
                expect(syncTime).toBeLessThanOrEqual(LATENCY_THRESHOLD);
                expect(fleetState.stateVersion).toBeGreaterThan(0);

                performanceMetrics.convergenceTime = syncTime;
            }, { retries: RETRY_ATTEMPTS });
        }, TEST_TIMEOUT);

        it('should maintain mesh network quality under load', async () => {
            const topology = await fleetManager.validateMeshTopology();
            const networkStats = await fleetManager.measureNetworkLatency();

            expect(topology.health).toBeGreaterThanOrEqual(0.95);
            expect(networkStats.averageLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
            expect(networkStats.packetLoss).toBeLessThanOrEqual(0.01);

            performanceMetrics.meshQuality = topology.health;
            performanceMetrics.latency = networkStats.averageLatency;
        });

        it('should handle member disconnections and reconnections gracefully', async () => {
            const fleet = await fleetManager.createFleet(testFleet);
            const testMember = fleet.members[0];

            // Test disconnection
            await fleetManager.leaveFleet(testMember.id);
            let fleetState = fleetManager.getFleetState();
            expect(fleetState.members).not.toContainEqual(expect.objectContaining({ id: testMember.id }));

            // Test reconnection
            await fleetManager.joinFleet(testMember.id, testMember.capabilities);
            fleetState = fleetManager.getFleetState();
            expect(fleetState.members).toContainEqual(expect.objectContaining({ id: testMember.id }));
        });
    });

    describe('Performance and Reliability', () => {
        it('should maintain real-time synchronization within latency bounds', async () => {
            const fleet = await fleetManager.createFleet(testFleet);
            const metrics = new Array<number>();

            // Measure sync latency over multiple operations
            for (let i = 0; i < 10; i++) {
                const startTime = Date.now();
                await fleetManager.synchronizeFleet({
                    documentId: fleet.id,
                    operation: 'UPDATE',
                    timestamp: Date.now(),
                    retryCount: 0
                });
                metrics.push(Date.now() - startTime);
            }

            const averageLatency = metrics.reduce((a, b) => a + b) / metrics.length;
            expect(averageLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
            performanceMetrics.latency = averageLatency;
        });

        it('should handle concurrent state updates without conflicts', async () => {
            const fleet = await fleetManager.createFleet(testFleet);
            const updatePromises = fleet.members.map(member => 
                fleetManager.synchronizeFleet({
                    documentId: fleet.id,
                    operation: 'UPDATE',
                    timestamp: Date.now(),
                    retryCount: 0
                })
            );

            await expect(Promise.all(updatePromises)).resolves.not.toThrow();
            const fleetState = fleetManager.getFleetState();
            expect(fleetState.stateVersion).toBe(fleet.members.length);
        });

        it('should maintain performance metrics within thresholds', async () => {
            const metrics = fleetManager.getMetrics();
            
            expect(metrics.averageLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
            expect(metrics.syncSuccessRate).toBeGreaterThanOrEqual(0.95);
            expect(metrics.memberCount).toBeLessThanOrEqual(MAX_TEST_FLEET_SIZE);
            expect(Date.now() - metrics.lastUpdateTimestamp).toBeLessThanOrEqual(SYNC_TEST_INTERVAL);
        });
    });
});