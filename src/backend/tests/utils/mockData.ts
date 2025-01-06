import { faker } from '@faker-js/faker'; // v8.0.0
import { v4 as uuidv4 } from 'uuid'; // v9.0.0

import { 
    IFleet, 
    IFleetMember, 
    INetworkStats,
    FleetRole,
    FleetStatus,
    MeshTopologyType
} from '../../src/interfaces/fleet.interface';

import {
    IGameState,
    IPerformanceMetrics,
    Vector3,
    IEnvironmentState,
    IPhysicsState
} from '../../src/interfaces/game.interface';

import {
    IPointCloud,
    IQualityMetrics,
    ScanQuality,
    ProcessingMode,
    IScanMetadata
} from '../../src/interfaces/lidar.interface';

// Constants for mock data generation
const DEFAULT_FLEET_SIZE = 32;
const DEFAULT_POINT_CLOUD_SIZE = 1_000_000;
const MAX_PROCESSING_TIME_MS = 50;
const MAX_SCAN_QUALITY = 1.0;
const MIN_CONFIDENCE_SCORE = 0.95;
const MAX_ERROR_RATE = 0.001;
const TARGET_FRAME_TIME_MS = 16.6;
const MAX_P2P_LATENCY_MS = 50;

/**
 * Generates a mock fleet with specified number of members and network metrics
 * @param memberCount Number of fleet members (max 32)
 * @param networkConfig Network performance configuration
 * @returns Mock fleet object
 */
export function generateMockFleet(
    memberCount: number = DEFAULT_FLEET_SIZE,
    networkConfig?: INetworkStats
): IFleet {
    const fleetId = uuidv4();
    const members: IFleetMember[] = [];

    // Generate fleet members
    for (let i = 0; i < Math.min(memberCount, DEFAULT_FLEET_SIZE); i++) {
        members.push({
            id: uuidv4(),
            deviceId: uuidv4(),
            role: i === 0 ? FleetRole.LEADER : FleetRole.MEMBER,
            status: FleetStatus.ACTIVE,
            joinedAt: Date.now() - faker.number.int({ min: 0, max: 3600000 }),
            lastActive: Date.now(),
            position: {
                x: faker.number.float({ min: -100, max: 100, precision: 0.01 }),
                y: faker.number.float({ min: -100, max: 100, precision: 0.01 }),
                z: faker.number.float({ min: -100, max: 100, precision: 0.01 }),
                timestamp: Date.now(),
                accuracy: faker.number.float({ min: 0.95, max: 1, precision: 0.001 })
            },
            capabilities: {
                lidarSupport: true,
                maxRange: 5.0,
                processingPower: faker.number.float({ min: 0.8, max: 1, precision: 0.01 }),
                networkBandwidth: faker.number.float({ min: 100, max: 1000, precision: 1 }),
                batteryLevel: faker.number.float({ min: 0.1, max: 1, precision: 0.01 })
            }
        });
    }

    return {
        id: fleetId,
        name: faker.word.words(2),
        maxDevices: DEFAULT_FLEET_SIZE,
        members,
        meshConfig: {
            topology: MeshTopologyType.HYBRID,
            maxPeers: DEFAULT_FLEET_SIZE,
            peerTimeout: MAX_P2P_LATENCY_MS,
            meshQuality: {
                connectionDensity: faker.number.float({ min: 0.8, max: 1, precision: 0.01 }),
                redundancyFactor: faker.number.float({ min: 1.5, max: 2.5, precision: 0.1 }),
                meshStability: faker.number.float({ min: 0.9, max: 1, precision: 0.01 }),
                routingEfficiency: faker.number.float({ min: 0.9, max: 1, precision: 0.01 })
            }
        },
        networkStats: networkConfig || {
            averageLatency: faker.number.float({ min: 20, max: MAX_P2P_LATENCY_MS, precision: 1 }),
            peakLatency: faker.number.float({ min: 30, max: MAX_P2P_LATENCY_MS, precision: 1 }),
            packetLoss: faker.number.float({ min: 0, max: 0.01, precision: 0.001 }),
            connectionQuality: faker.number.float({ min: 0.95, max: 1, precision: 0.01 }),
            meshHealth: faker.number.float({ min: 0.95, max: 1, precision: 0.01 }),
            lastUpdate: Date.now()
        },
        createdAt: Date.now() - 3600000,
        lastUpdated: Date.now()
    };
}

/**
 * Generates mock game state with performance metrics
 * @param perfConfig Performance metrics configuration
 * @returns Mock game state object
 */
export function generateMockGameState(perfConfig?: IPerformanceMetrics): IGameState {
    const gameId = uuidv4();
    const sessionId = uuidv4();
    const timestamp = Date.now();

    const environment: IEnvironmentState = {
        timestamp,
        scanQuality: faker.number.float({ min: 0.95, max: 1, precision: 0.01 }),
        pointCount: DEFAULT_POINT_CLOUD_SIZE,
        classifiedObjects: Array.from({ length: 10 }, () => ({
            id: uuidv4(),
            type: faker.helpers.arrayElement(['wall', 'floor', 'object', 'player']),
            position: generateMockVector3(),
            dimensions: generateMockVector3(),
            confidence: faker.number.float({ min: MIN_CONFIDENCE_SCORE, max: 1, precision: 0.01 }),
            timestamp
        })),
        lidarMetrics: {
            scanRate: 30,
            resolution: 0.01,
            effectiveRange: 5.0,
            pointDensity: faker.number.float({ min: 1000, max: 2000, precision: 1 })
        }
    };

    const physics: IPhysicsState = {
        timestamp,
        objects: Array.from({ length: 5 }, () => ({
            id: uuidv4(),
            position: generateMockVector3(),
            velocity: generateMockVector3(),
            mass: faker.number.float({ min: 0.1, max: 100, precision: 0.1 }),
            collisionMesh: new ArrayBuffer(1024),
            lastUpdateTimestamp: timestamp
        })),
        collisions: [],
        simulationLatency: faker.number.float({ min: 1, max: 5, precision: 0.1 })
    };

    return {
        gameId,
        sessionId,
        fleetId: uuidv4(),
        deviceCount: faker.number.int({ min: 1, max: DEFAULT_FLEET_SIZE }),
        timestamp,
        environment,
        physics,
        metrics: perfConfig || {
            stateUpdateLatency: faker.number.float({ min: 1, max: 10, precision: 0.1 }),
            lidarProcessingLatency: faker.number.float({ min: 20, max: MAX_PROCESSING_TIME_MS, precision: 0.1 }),
            physicsSimulationLatency: faker.number.float({ min: 1, max: 5, precision: 0.1 }),
            fleetSyncLatency: faker.number.float({ min: 10, max: MAX_P2P_LATENCY_MS, precision: 0.1 })
        }
    };
}

/**
 * Generates mock point cloud data with quality metrics
 * @param pointCount Number of points in cloud
 * @param qualityConfig Quality metrics configuration
 * @returns Mock point cloud object
 */
export function generateMockPointCloud(
    pointCount: number = DEFAULT_POINT_CLOUD_SIZE,
    qualityConfig?: IQualityMetrics
): IPointCloud {
    // Generate mock point cloud buffer
    const points = Buffer.alloc(pointCount * 12); // 3 float32 values per point (x,y,z)
    for (let i = 0; i < pointCount; i++) {
        points.writeFloatLE(faker.number.float({ min: -5, max: 5, precision: 0.01 }), i * 12);
        points.writeFloatLE(faker.number.float({ min: -5, max: 5, precision: 0.01 }), i * 12 + 4);
        points.writeFloatLE(faker.number.float({ min: -5, max: 5, precision: 0.01 }), i * 12 + 8);
    }

    return {
        points,
        timestamp: Date.now(),
        quality: ScanQuality.HIGH,
        density: faker.number.float({ min: 1000, max: 2000, precision: 1 }),
        confidence: faker.number.float({ min: MIN_CONFIDENCE_SCORE, max: 1, precision: 0.001 })
    };
}

/**
 * Generates mock scan metadata with performance metrics
 * @returns Mock scan metadata object
 */
export function generateMockScanMetadata(): IScanMetadata {
    return {
        scanId: uuidv4(),
        timestamp: Date.now(),
        processingTime: faker.number.float({ min: 20, max: MAX_PROCESSING_TIME_MS, precision: 0.1 }),
        quality: ScanQuality.HIGH,
        errorRate: faker.number.float({ min: 0, max: MAX_ERROR_RATE, precision: 0.0001 }),
        powerConsumption: faker.number.float({ min: 1, max: 5, precision: 0.1 })
    };
}

/**
 * Helper function to generate mock 3D vector
 * @returns Mock Vector3 object
 */
function generateMockVector3(): Vector3 {
    return {
        x: faker.number.float({ min: -100, max: 100, precision: 0.01 }),
        y: faker.number.float({ min: -100, max: 100, precision: 0.01 }),
        z: faker.number.float({ min: -100, max: 100, precision: 0.01 })
    };
}