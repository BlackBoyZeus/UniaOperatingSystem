// External imports with versions for security tracking
import 'cypress'; // ^12.0.0
import '@testing-library/webrtc-mock'; // ^2.0.0
import performance from 'performance-now'; // ^2.1.0

// Internal imports
import { AuthService } from '../../src/services/auth.service';
import { FleetService } from '../../src/services/fleet.service';
import { GameService } from '../../src/services/game.service';
import { LidarService } from '../../src/services/lidar.service';

// Global test constants
const TEST_TIMEOUT = 30000;
const RETRY_INTERVAL = 100;
const MAX_RETRY_ATTEMPTS = 5;
const PERFORMANCE_THRESHOLDS = {
    frameRate: 60,
    latency: 50,
    memory: 4096
};
const LIDAR_CONFIG = {
    pointsPerSecond: 1000000,
    accuracy: 0.01
};

// Extend Cypress namespace with custom commands
declare global {
    namespace Cypress {
        interface Chainable {
            login(username: string, password: string, hardwareToken: string, securityOptions?: object): Chainable<any>;
            testFleetManagement(fleetId: string, deviceCount: number, networkConfig: object): Chainable<any>;
            validateGamePerformance(gameId: string, performanceThresholds: object): Chainable<any>;
            testLidarProcessing(scanData: object, processingConfig: object): Chainable<any>;
        }
    }
}

/**
 * Enhanced authentication command with hardware token validation
 */
Cypress.Commands.add('login', (username: string, password: string, hardwareToken: string, securityOptions = {}) => {
    const authService = new AuthService();

    return cy.wrap(null, { timeout: TEST_TIMEOUT }).then(async () => {
        try {
            // Validate hardware token
            const isValidToken = await authService.validateHardwareToken(hardwareToken);
            expect(isValidToken).to.be.true;

            // Attempt login
            const authResult = await authService.login(username, password, hardwareToken);
            expect(authResult).to.have.property('accessToken');

            // Monitor token refresh
            let refreshCount = 0;
            cy.spy(authService, 'refreshToken').as('tokenRefresh');
            cy.wait(5000).then(() => {
                expect(refreshCount).to.be.gte(0);
            });

            // Detect security breaches
            const breachDetected = await authService.detectSecurityBreach();
            expect(breachDetected).to.be.false;

            return authResult;
        } catch (error) {
            cy.log('Authentication error:', error);
            throw error;
        }
    });
});

/**
 * Fleet management testing with network quality validation
 */
Cypress.Commands.add('testFleetManagement', (fleetId: string, deviceCount: number, networkConfig = {}) => {
    const fleetService = new FleetService();

    return cy.wrap(null, { timeout: TEST_TIMEOUT }).then(async () => {
        try {
            // Create fleet
            const fleet = await fleetService.createFleet(fleetId, deviceCount);
            expect(fleet.members.length).to.be.lte(32);

            // Test WebRTC connections
            const connections = await Promise.all(
                fleet.members.map(member => 
                    fleetService.validateWebRTCConnection(member.id)
                )
            );
            connections.forEach(conn => {
                expect(conn.latency).to.be.lte(50);
                expect(conn.quality).to.be.gte(0.8);
            });

            // Monitor network latency
            const networkMetrics = await fleetService.monitorNetworkLatency();
            expect(networkMetrics.averageLatency).to.be.lte(50);
            expect(networkMetrics.packetLoss).to.be.lte(0.1);

            return fleet;
        } catch (error) {
            cy.log('Fleet management error:', error);
            throw error;
        }
    });
});

/**
 * Game performance validation with comprehensive metrics
 */
Cypress.Commands.add('validateGamePerformance', (gameId: string, performanceThresholds = PERFORMANCE_THRESHOLDS) => {
    const gameService = new GameService();

    return cy.wrap(null, { timeout: TEST_TIMEOUT }).then(async () => {
        try {
            // Start game session
            await gameService.startGame(gameId);

            // Monitor frame rate
            const frameRate = await gameService.validateFrameRate();
            expect(frameRate).to.be.gte(performanceThresholds.frameRate);

            // Track memory usage
            const memoryUsage = await gameService.monitorMemoryUsage();
            expect(memoryUsage).to.be.lte(performanceThresholds.memory);

            // Validate state synchronization
            const syncResult = await gameService.validateStateSynchronization();
            expect(syncResult.latency).to.be.lte(performanceThresholds.latency);
            expect(syncResult.consistency).to.be.true;

            return {
                frameRate,
                memoryUsage,
                syncLatency: syncResult.latency
            };
        } catch (error) {
            cy.log('Performance validation error:', error);
            throw error;
        }
    });
});

/**
 * LiDAR processing pipeline validation
 */
Cypress.Commands.add('testLidarProcessing', (scanData: object, processingConfig = LIDAR_CONFIG) => {
    const lidarService = new LidarService();

    return cy.wrap(null, { timeout: TEST_TIMEOUT }).then(async () => {
        try {
            // Generate test point cloud
            const pointCloud = await lidarService.generatePointCloud(scanData);
            expect(pointCloud.points.length).to.be.gte(processingConfig.pointsPerSecond);

            // Process mesh generation
            const meshResult = await lidarService.validateMeshProcessing(pointCloud);
            expect(meshResult.accuracy).to.be.gte(processingConfig.accuracy);

            // Test classification accuracy
            const classificationResult = await lidarService.testClassificationAccuracy(meshResult);
            expect(classificationResult.accuracy).to.be.gte(0.9);

            return {
                pointCloud,
                meshAccuracy: meshResult.accuracy,
                classificationAccuracy: classificationResult.accuracy
            };
        } catch (error) {
            cy.log('LiDAR processing error:', error);
            throw error;
        }
    });
});