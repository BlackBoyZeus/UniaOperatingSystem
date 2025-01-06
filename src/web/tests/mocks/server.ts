import { setupServer } from 'msw/node'; // v1.2.0
import { rest } from 'msw'; // v1.2.0
import * as Automerge from 'automerge'; // v2.0.0
import { 
    createFleetHandler, 
    joinFleetHandler, 
    gameStateHandler, 
    lidarDataHandler, 
    webrtcSignalingHandler 
} from './handlers';
import { FleetStatus, FleetRole } from '../../src/types/fleet.types';
import type { IFleet } from '../../src/interfaces/fleet.interface';

// Global constants for mock server configuration
export const MOCK_API_URL = 'http://localhost:3000';
export const MOCK_WEBSOCKET_URL = 'ws://localhost:3000';
export const MOCK_WEBRTC_URL = 'wss://localhost:3000/webrtc';
export const DEFAULT_LATENCY = 45; // Target network latency in ms
export const MAX_FLEET_SIZE = 32;

// Network condition simulation interface
interface NetworkProfile {
    latency: number;
    jitter: number;
    packetLoss: number;
    bandwidth: number;
}

// Default network profile
const DEFAULT_NETWORK_PROFILE: NetworkProfile = {
    latency: DEFAULT_LATENCY,
    jitter: 5, // ms of random variation
    packetLoss: 0.01, // 1% packet loss
    bandwidth: 1000000 // 1 Mbps
};

// Telemetry collection for test analysis
interface ServerTelemetry {
    requestCount: number;
    avgLatency: number;
    errorRate: number;
    activeFleets: Map<string, IFleet>;
    networkStats: {
        totalBytesTransferred: number;
        packetsLost: number;
        avgBandwidth: number;
    };
}

let telemetry: ServerTelemetry = {
    requestCount: 0,
    avgLatency: DEFAULT_LATENCY,
    errorRate: 0,
    activeFleets: new Map(),
    networkStats: {
        totalBytesTransferred: 0,
        packetsLost: 0,
        avgBandwidth: 0
    }
};

/**
 * Simulates network conditions for response delays and errors
 * @param profile Network condition profile to simulate
 */
const simulateNetworkConditions = (profile: NetworkProfile = DEFAULT_NETWORK_PROFILE) => 
    async (req: any, res: any, ctx: any) => {
        // Simulate packet loss
        if (Math.random() < profile.packetLoss) {
            telemetry.networkStats.packetsLost++;
            throw new Error('Simulated packet loss');
        }

        // Add latency with jitter
        const delay = profile.latency + (Math.random() * profile.jitter * 2 - profile.jitter);
        await new Promise(resolve => setTimeout(resolve, delay));

        // Update telemetry
        telemetry.avgLatency = (telemetry.avgLatency * telemetry.requestCount + delay) / 
            (telemetry.requestCount + 1);
        telemetry.requestCount++;

        return res;
    };

/**
 * Creates and configures MSW server instance with enhanced features
 * @param networkProfile Optional network simulation profile
 * @returns Configured MSW server instance
 */
const createMockServer = (networkProfile: NetworkProfile = DEFAULT_NETWORK_PROFILE) => {
    // Initialize CRDT document for fleet state
    const fleetDoc = Automerge.init<any>();

    // Create server instance with handlers
    const server = setupServer(
        // Fleet management endpoints
        rest.post(
            `${MOCK_API_URL}/fleet/create`,
            simulateNetworkConditions(networkProfile),
            createFleetHandler(fleetDoc)
        ),
        rest.post(
            `${MOCK_API_URL}/fleet/join`,
            simulateNetworkConditions(networkProfile),
            joinFleetHandler(fleetDoc)
        ),
        rest.post(
            `${MOCK_API_URL}/game/sync`,
            simulateNetworkConditions(networkProfile),
            gameStateHandler(fleetDoc)
        ),
        rest.post(
            `${MOCK_API_URL}/lidar/upload`,
            simulateNetworkConditions(networkProfile),
            lidarDataHandler
        ),
        rest.ws(
            `${MOCK_WEBRTC_URL}`,
            webrtcSignalingHandler(networkProfile)
        )
    );

    // Enhance server with additional features
    return {
        ...server,
        setNetworkProfile: (profile: NetworkProfile) => {
            Object.assign(networkProfile, profile);
        },
        getTelemetry: () => ({ ...telemetry }),
        resetTelemetry: () => {
            telemetry = {
                requestCount: 0,
                avgLatency: DEFAULT_LATENCY,
                errorRate: 0,
                activeFleets: new Map(),
                networkStats: {
                    totalBytesTransferred: 0,
                    packetsLost: 0,
                    avgBandwidth: 0
                }
            };
        }
    };
};

// Create and export server instance
export const server = createMockServer();

// Export type definitions for consumers
export type MockServer = ReturnType<typeof createMockServer>;
export type { NetworkProfile, ServerTelemetry };