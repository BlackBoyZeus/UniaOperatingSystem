import { rest } from 'msw';
import * as automerge from 'automerge';
import { IFleet, IFleetMember } from '../../src/interfaces/fleet.interface';
import { apiConfig } from '../../src/config/api.config';
import { 
    FleetStatus, 
    FleetRole, 
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD,
    FleetMessageType 
} from '../../src/types/fleet.types';

// Global state for mock fleet management
const MOCK_FLEET_STATE = new Map<string, IFleet & { doc: automerge.Doc<any> }>();
const MOCK_LATENCY = 45; // Simulated network latency in ms
const MOCK_NETWORK_CONDITIONS = {
    jitter: 5, // ms of random latency variation
    packetLoss: 0.01, // 1% packet loss rate
    bandwidth: 1000000 // 1 Mbps bandwidth limit
};

// Utility decorators for network simulation
const withLatencySimulation = (handler: Function) => async (...args: any[]) => {
    await new Promise(resolve => 
        setTimeout(resolve, MOCK_LATENCY + (Math.random() * MOCK_NETWORK_CONDITIONS.jitter))
    );
    return handler(...args);
};

const withNetworkConditions = (handler: Function) => async (...args: any[]) => {
    if (Math.random() < MOCK_NETWORK_CONDITIONS.packetLoss) {
        throw new Error('Simulated packet loss');
    }
    return handler(...args);
};

// Mock request handlers
export const handlers = [
    // Fleet Creation Handler
    rest.post(apiConfig.endpoints.FLEET.CREATE, async (req, res, ctx) => {
        const { name, maxDevices } = await req.json();
        
        if (maxDevices > MAX_FLEET_SIZE) {
            return res(
                ctx.status(400),
                ctx.json({ error: `Fleet size cannot exceed ${MAX_FLEET_SIZE} devices` })
            );
        }

        const fleetId = crypto.randomUUID();
        const initialDoc = automerge.init();
        
        const fleet: IFleet = {
            id: fleetId,
            name,
            maxDevices,
            members: [],
            status: FleetStatus.ACTIVE,
            networkStats: {
                averageLatency: MOCK_LATENCY,
                maxLatency: MOCK_LATENCY + MOCK_NETWORK_CONDITIONS.jitter,
                minLatency: MOCK_LATENCY - MOCK_NETWORK_CONDITIONS.jitter,
                packetsLost: 0,
                bandwidth: MOCK_NETWORK_CONDITIONS.bandwidth,
                connectedPeers: 0,
                syncLatency: MOCK_LATENCY
            },
            qualityMetrics: {
                connectionScore: 1,
                syncSuccess: 100,
                leaderRedundancy: 1
            },
            backupLeaders: []
        };

        MOCK_FLEET_STATE.set(fleetId, { ...fleet, doc: initialDoc });

        return res(
            ctx.delay(MOCK_LATENCY),
            ctx.status(201),
            ctx.json(fleet)
        );
    }),

    // Fleet Join Handler
    rest.post(apiConfig.endpoints.FLEET.JOIN, withLatencySimulation(async (req, res, ctx) => {
        const { fleetId, deviceId } = await req.json();
        const fleet = MOCK_FLEET_STATE.get(fleetId);

        if (!fleet) {
            return res(
                ctx.status(404),
                ctx.json({ error: 'Fleet not found' })
            );
        }

        if (fleet.members.length >= fleet.maxDevices) {
            return res(
                ctx.status(400),
                ctx.json({ error: 'Fleet is at maximum capacity' })
            );
        }

        const member: IFleetMember = {
            id: crypto.randomUUID(),
            deviceId,
            role: fleet.members.length === 0 ? FleetRole.LEADER : FleetRole.MEMBER,
            connection: {
                lastPing: Date.now(),
                connectionQuality: 1,
                retryCount: 0
            },
            latency: MOCK_LATENCY,
            connectionQuality: {
                signalStrength: 0.95,
                stability: 0.98,
                reliability: 0.99
            },
            lastCRDTOperation: {
                timestamp: Date.now(),
                type: FleetMessageType.MEMBER_JOIN,
                payload: null
            }
        };

        fleet.members.push(member);
        fleet.networkStats.connectedPeers = fleet.members.length;

        return res(
            ctx.status(200),
            ctx.json({ member, fleet })
        );
    })),

    // Game State Sync Handler
    rest.post(apiConfig.endpoints.GAME.SYNC, withNetworkConditions(async (req, res, ctx) => {
        const { fleetId, memberId, state } = await req.json();
        const fleet = MOCK_FLEET_STATE.get(fleetId);

        if (!fleet) {
            return res(
                ctx.status(404),
                ctx.json({ error: 'Fleet not found' })
            );
        }

        const newDoc = automerge.change(fleet.doc, doc => {
            doc.gameState = { ...doc.gameState, ...state };
        });

        fleet.doc = newDoc;

        return res(
            ctx.status(200),
            ctx.json({
                state: automerge.getLastLocalChange(newDoc),
                timestamp: Date.now(),
                syncLatency: MOCK_LATENCY
            })
        );
    })),

    // LiDAR Data Handler
    rest.post(apiConfig.endpoints.LIDAR.UPLOAD, async (req, res, ctx) => {
        const chunks: Uint8Array[] = [];
        const reader = req.body?.getReader();
        
        if (!reader) {
            return res(
                ctx.status(400),
                ctx.json({ error: 'Invalid request body' })
            );
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            
            // Simulate processing delay based on chunk size
            await new Promise(resolve => 
                setTimeout(resolve, Math.floor(value.length / MOCK_NETWORK_CONDITIONS.bandwidth * 1000))
            );
        }

        return res(
            ctx.delay(MOCK_LATENCY),
            ctx.status(200),
            ctx.json({
                bytesProcessed: chunks.reduce((acc, chunk) => acc + chunk.length, 0),
                processingTime: MOCK_LATENCY,
                quality: 0.95
            })
        );
    })
];

export default handlers;