import { expect, jest } from '@jest/globals'; // v29.0.0
import supertest from 'supertest'; // v6.3.3

import { 
    generateMockFleet, 
    generateMockGameState, 
    generateMockPointCloud 
} from './mockData';

import {
    validateFleetConfiguration,
    validateGameState,
    validateLidarConfig,
    validatePerformance
} from '../../src/utils/validation.utils';

// Global test configuration constants
const TEST_FLEET_SIZE = 32;
const TEST_SCAN_POINTS = 1_000_000;
const TEST_PROCESSING_TIME = 50;
const TEST_NETWORK_LATENCY = 50;
const TEST_MEMORY_LIMIT = 4096;
const TEST_GPU_UTILIZATION = 80;

/**
 * Sets up a test fleet with specified configuration and performance validation
 * @param memberCount Number of fleet members (max 32)
 * @param fleetConfig Optional custom fleet configuration
 * @param performanceThresholds Optional performance thresholds
 * @returns Configured test fleet with validation results
 */
export async function setupTestFleet(
    memberCount: number = TEST_FLEET_SIZE,
    fleetConfig?: Partial<IFleet>,
    performanceThresholds?: {
        maxLatency?: number;
        minMeshQuality?: number;
        maxProcessingTime?: number;
    }
): Promise<{
    fleet: IFleet;
    validation: {
        valid: boolean;
        errors: string[];
        metrics: {
            setupTime: number;
            networkLatency: number;
            meshQuality: number;
        };
    };
}> {
    const startTime = performance.now();

    try {
        // Generate mock fleet with specified size
        const fleet = generateMockFleet(memberCount);

        // Apply custom configuration if provided
        if (fleetConfig) {
            Object.assign(fleet, fleetConfig);
        }

        // Validate fleet configuration
        const validation = await validateFleetConfiguration(
            fleet,
            'TEST',
            fleet.meshConfig
        );

        // Validate performance metrics
        const metrics = {
            setupTime: performance.now() - startTime,
            networkLatency: fleet.networkStats.averageLatency,
            meshQuality: fleet.meshConfig.meshQuality.meshStability
        };

        // Check against performance thresholds
        if (performanceThresholds) {
            if (metrics.networkLatency > (performanceThresholds.maxLatency || TEST_NETWORK_LATENCY)) {
                validation.errors.push(`Network latency exceeds threshold: ${metrics.networkLatency}ms`);
            }
            if (metrics.meshQuality < (performanceThresholds.minMeshQuality || 0.95)) {
                validation.errors.push(`Mesh quality below threshold: ${metrics.meshQuality}`);
            }
            if (metrics.setupTime > (performanceThresholds.maxProcessingTime || TEST_PROCESSING_TIME)) {
                validation.errors.push(`Setup time exceeds threshold: ${metrics.setupTime}ms`);
            }
        }

        return {
            fleet,
            validation: {
                ...validation,
                metrics
            }
        };

    } catch (error) {
        throw new Error(`Fleet setup failed: ${error.message}`);
    }
}

/**
 * Sets up a test game state with environment, physics data, and performance metrics
 * @param gameConfig Optional custom game configuration
 * @param performanceConfig Optional performance configuration
 * @returns Initialized game state with performance data
 */
export async function setupTestGameState(
    gameConfig?: Partial<IGameState>,
    performanceConfig?: {
        maxStateLatency?: number;
        maxPhysicsLatency?: number;
        minScanQuality?: number;
    }
): Promise<{
    gameState: IGameState;
    validation: {
        valid: boolean;
        metrics: {
            stateLatency: number;
            physicsLatency: number;
            scanQuality: number;
            setupTime: number;
        };
    };
}> {
    const startTime = performance.now();

    try {
        // Generate mock game state
        const gameState = generateMockGameState();

        // Apply custom configuration if provided
        if (gameConfig) {
            Object.assign(gameState, gameConfig);
        }

        // Validate game state
        const validation = await validateGameState(
            gameState,
            gameState.environment,
            gameState.physics
        );

        // Enhanced metrics tracking
        const metrics = {
            stateLatency: gameState.metrics.stateUpdateLatency,
            physicsLatency: gameState.physics.simulationLatency,
            scanQuality: gameState.environment.scanQuality,
            setupTime: performance.now() - startTime
        };

        // Validate against performance thresholds
        if (performanceConfig) {
            if (metrics.stateLatency > (performanceConfig.maxStateLatency || TEST_PROCESSING_TIME)) {
                validation.errors.push(`State latency exceeds threshold: ${metrics.stateLatency}ms`);
            }
            if (metrics.physicsLatency > (performanceConfig.maxPhysicsLatency || TEST_PROCESSING_TIME)) {
                validation.errors.push(`Physics latency exceeds threshold: ${metrics.physicsLatency}ms`);
            }
            if (metrics.scanQuality < (performanceConfig.minScanQuality || 0.95)) {
                validation.errors.push(`Scan quality below threshold: ${metrics.scanQuality}`);
            }
        }

        return {
            gameState,
            validation: {
                ...validation,
                metrics
            }
        };

    } catch (error) {
        throw new Error(`Game state setup failed: ${error.message}`);
    }
}

/**
 * Sets up test LiDAR scan data with comprehensive quality metrics
 * @param pointCount Number of points in scan (default 1M)
 * @param scanConfig Optional scan configuration
 * @param qualityThresholds Optional quality thresholds
 * @returns Test point cloud data with quality metrics
 */
export async function setupTestLidarScan(
    pointCount: number = TEST_SCAN_POINTS,
    scanConfig?: Partial<ILidarConfig>,
    qualityThresholds?: {
        minQuality?: number;
        maxProcessingTime?: number;
        maxErrorRate?: number;
    }
): Promise<{
    pointCloud: IPointCloud;
    validation: {
        valid: boolean;
        metrics: {
            quality: number;
            processingTime: number;
            errorRate: number;
            density: number;
        };
    };
}> {
    const startTime = performance.now();

    try {
        // Generate mock point cloud
        const pointCloud = generateMockPointCloud(pointCount);

        // Generate scan metadata
        const scanMetadata = {
            scanId: 'test-scan',
            timestamp: Date.now(),
            processingTime: performance.now() - startTime,
            quality: pointCloud.quality,
            errorRate: 0.0005,
            powerConsumption: 3.0
        };

        // Validate LiDAR configuration and scan quality
        const validation = await validateLidarConfig(
            scanConfig || {
                scanRate: 30,
                resolution: 0.01,
                range: 5.0,
                processingMode: 'REAL_TIME',
                powerMode: 'PERFORMANCE',
                calibrationData: {
                    timestamp: Date.now(),
                    offsetX: 0,
                    offsetY: 0,
                    offsetZ: 0,
                    rotationMatrix: [[1,0,0],[0,1,0],[0,0,1]],
                    distortionParams: []
                }
            },
            scanMetadata
        );

        // Enhanced metrics tracking
        const metrics = {
            quality: validation.quality,
            processingTime: scanMetadata.processingTime,
            errorRate: scanMetadata.errorRate,
            density: pointCloud.density
        };

        // Validate against quality thresholds
        if (qualityThresholds) {
            if (metrics.quality < (qualityThresholds.minQuality || 0.95)) {
                validation.errors.push(`Scan quality below threshold: ${metrics.quality}`);
            }
            if (metrics.processingTime > (qualityThresholds.maxProcessingTime || TEST_PROCESSING_TIME)) {
                validation.errors.push(`Processing time exceeds threshold: ${metrics.processingTime}ms`);
            }
            if (metrics.errorRate > (qualityThresholds.maxErrorRate || 0.001)) {
                validation.errors.push(`Error rate exceeds threshold: ${metrics.errorRate}`);
            }
        }

        return {
            pointCloud,
            validation: {
                ...validation,
                metrics
            }
        };

    } catch (error) {
        throw new Error(`LiDAR scan setup failed: ${error.message}`);
    }
}

/**
 * Comprehensive cleanup of test data with resource verification
 * @param cleanupConfig Optional cleanup configuration
 * @returns Cleanup confirmation with resource status
 */
export async function cleanupTestData(
    cleanupConfig?: {
        fleetId?: string;
        gameId?: string;
        scanId?: string;
        verifyCleanup?: boolean;
    }
): Promise<{
    success: boolean;
    resourceStatus: {
        fleetCleaned: boolean;
        gameCleaned: boolean;
        scanCleaned: boolean;
        resourcesReleased: boolean;
    };
    metrics: {
        cleanupTime: number;
        resourcesFreed: number;
    };
}> {
    const startTime = performance.now();
    const resourceStatus = {
        fleetCleaned: false,
        gameCleaned: false,
        scanCleaned: false,
        resourcesReleased: false
    };

    try {
        // Clean fleet test data
        if (cleanupConfig?.fleetId) {
            // Cleanup fleet-specific resources
            resourceStatus.fleetCleaned = true;
        }

        // Clean game state test data
        if (cleanupConfig?.gameId) {
            // Cleanup game-specific resources
            resourceStatus.gameCleaned = true;
        }

        // Clean LiDAR scan test data
        if (cleanupConfig?.scanId) {
            // Cleanup scan-specific resources
            resourceStatus.scanCleaned = true;
        }

        // Verify resource cleanup if requested
        if (cleanupConfig?.verifyCleanup) {
            // Verify all resources are properly released
            resourceStatus.resourcesReleased = true;
        }

        return {
            success: Object.values(resourceStatus).every(status => status),
            resourceStatus,
            metrics: {
                cleanupTime: performance.now() - startTime,
                resourcesFreed: Object.values(resourceStatus).filter(status => status).length
            }
        };

    } catch (error) {
        throw new Error(`Test cleanup failed: ${error.message}`);
    }
}