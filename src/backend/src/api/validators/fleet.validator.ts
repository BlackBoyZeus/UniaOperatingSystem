import { z } from 'zod'; // v3.22.2
import { 
    IFleet, IFleetMember, IMeshConfig, IWebRTCConfig, ICRDTConfig,
} from '../../interfaces/fleet.interface';
import {
    validateFleetConfiguration,
    validateWebRTCConfig,
    validateCRDTState
} from '../../utils/validation.utils';

// Global validation constants
export const MAX_FLEET_SIZE = 32;
export const MAX_LATENCY_MS = 50;
export const MESH_TOPOLOGY_TYPES = ['full', 'star', 'ring'] as const;
export const FLEET_ROLES = ['leader', 'member'] as const;
export const FLEET_STATUS = ['active', 'inactive', 'connecting'] as const;
export const WEBRTC_MODES = ['p2p', 'relay'] as const;
export const CRDT_TYPES = ['automerge', 'yjs'] as const;
export const VALIDATION_CACHE_TTL = 300000; // 5 minutes

// WebRTC configuration schema
export const webRTCConfigSchema = z.object({
    mode: z.enum(WEBRTC_MODES),
    iceServers: z.array(z.object({
        urls: z.array(z.string().url()),
        username: z.string().optional(),
        credential: z.string().optional()
    })),
    maxLatency: z.number().max(MAX_LATENCY_MS),
    reconnectStrategy: z.object({
        maxAttempts: z.number().min(1),
        backoffMultiplier: z.number().min(1),
        initialDelay: z.number().min(100),
        maxDelay: z.number().max(30000)
    })
});

// CRDT configuration schema
export const crdtConfigSchema = z.object({
    type: z.enum(CRDT_TYPES),
    syncInterval: z.number().min(10).max(1000),
    conflictResolution: z.enum(['lastWriteWins', 'customMerge']),
    customMergeFunction: z.function().optional(),
    stateValidation: z.boolean(),
    maxStateSize: z.number().max(1024 * 1024) // 1MB max state size
});

// Mesh network configuration schema
export const meshConfigSchema = z.object({
    topology: z.enum(MESH_TOPOLOGY_TYPES),
    maxPeers: z.number().max(MAX_FLEET_SIZE),
    reconnectStrategy: z.object({
        maxAttempts: z.number().min(1),
        backoffMultiplier: z.number().min(1),
        initialDelay: z.number().min(100),
        maxDelay: z.number().max(30000)
    }),
    peerTimeout: z.number().min(1000).max(30000),
    signalServer: z.string().url(),
    iceServers: z.array(z.object({
        urls: z.array(z.string().url()),
        username: z.string().optional(),
        credential: z.string().optional()
    })),
    meshQuality: z.object({
        connectionDensity: z.number().min(0).max(1),
        redundancyFactor: z.number().min(1),
        meshStability: z.number().min(0).max(1),
        routingEfficiency: z.number().min(0).max(1)
    })
});

// Fleet member schema
export const fleetMemberSchema = z.object({
    id: z.string().uuid(),
    deviceId: z.string(),
    role: z.enum(FLEET_ROLES),
    status: z.enum(FLEET_STATUS),
    joinedAt: z.number(),
    lastActive: z.number(),
    position: z.object({
        x: z.number(),
        y: z.number(),
        z: z.number(),
        timestamp: z.number(),
        accuracy: z.number()
    }),
    capabilities: z.object({
        lidarSupport: z.boolean(),
        maxRange: z.number().positive(),
        processingPower: z.number().positive(),
        networkBandwidth: z.number().positive(),
        batteryLevel: z.number().min(0).max(100)
    })
});

// Core fleet schema
export const fleetSchema = z.object({
    id: z.string().uuid(),
    name: z.string().min(3).max(50),
    maxDevices: z.number().max(MAX_FLEET_SIZE),
    members: z.array(fleetMemberSchema).max(MAX_FLEET_SIZE),
    meshConfig: meshConfigSchema,
    webRTCConfig: webRTCConfigSchema,
    crdtConfig: crdtConfigSchema,
    networkStats: z.object({
        averageLatency: z.number().max(MAX_LATENCY_MS),
        peakLatency: z.number(),
        packetLoss: z.number().min(0).max(1),
        bandwidth: z.object({
            current: z.number().positive(),
            peak: z.number().positive(),
            average: z.number().positive(),
            totalTransferred: z.number().positive(),
            lastMeasured: z.number()
        }),
        connectionQuality: z.number().min(0).max(1),
        meshHealth: z.number().min(0).max(1),
        lastUpdate: z.number()
    }),
    createdAt: z.number(),
    lastUpdated: z.number()
});

/**
 * Validates fleet creation request data against platform requirements
 * @param fleetData Fleet configuration data to validate
 * @returns Promise resolving to true if valid, throws ValidationError otherwise
 */
export async function validateFleetCreation(fleetData: IFleet): Promise<boolean> {
    try {
        // Parse and validate against core schema
        await fleetSchema.parseAsync(fleetData);

        // Validate fleet size constraints
        if (fleetData.members.length > MAX_FLEET_SIZE) {
            throw new Error(`Fleet size exceeds maximum limit of ${MAX_FLEET_SIZE} devices`);
        }

        // Validate mesh network configuration
        if (fleetData.meshConfig.topology === 'full' && fleetData.members.length > 16) {
            throw new Error('Full mesh topology limited to 16 devices for performance');
        }

        // Validate WebRTC configuration
        const webRTCValid = await validateWebRTCConfig(fleetData.webRTCConfig);
        if (!webRTCValid) {
            throw new Error('Invalid WebRTC configuration');
        }

        // Validate CRDT configuration
        const crdtValid = await validateCRDTState(fleetData.crdtConfig);
        if (!crdtValid) {
            throw new Error('Invalid CRDT configuration');
        }

        // Validate network performance requirements
        if (fleetData.networkStats.averageLatency > MAX_LATENCY_MS) {
            throw new Error(`Network latency exceeds maximum threshold of ${MAX_LATENCY_MS}ms`);
        }

        return true;
    } catch (error) {
        throw new Error(`Fleet validation error: ${error.message}`);
    }
}

/**
 * Validates fleet update request data with partial validation support
 * @param updateData Partial fleet update data
 * @returns Promise resolving to true if valid, throws ValidationError otherwise
 */
export async function validateFleetUpdate(updateData: Partial<IFleet>): Promise<boolean> {
    try {
        // Create partial schema for update validation
        const partialFleetSchema = fleetSchema.partial();
        await partialFleetSchema.parseAsync(updateData);

        // Validate size constraints if members are being updated
        if (updateData.members && updateData.members.length > MAX_FLEET_SIZE) {
            throw new Error(`Updated fleet size exceeds maximum limit of ${MAX_FLEET_SIZE} devices`);
        }

        // Validate mesh config changes if present
        if (updateData.meshConfig) {
            await meshConfigSchema.parseAsync(updateData.meshConfig);
        }

        // Validate WebRTC config changes if present
        if (updateData.webRTCConfig) {
            await webRTCConfigSchema.parseAsync(updateData.webRTCConfig);
        }

        // Validate CRDT config changes if present
        if (updateData.crdtConfig) {
            await crdtConfigSchema.parseAsync(updateData.crdtConfig);
        }

        // Validate network stats if present
        if (updateData.networkStats?.averageLatency) {
            if (updateData.networkStats.averageLatency > MAX_LATENCY_MS) {
                throw new Error(`Updated network latency exceeds maximum threshold of ${MAX_LATENCY_MS}ms`);
            }
        }

        return true;
    } catch (error) {
        throw new Error(`Fleet update validation error: ${error.message}`);
    }
}