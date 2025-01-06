import { z } from 'zod'; // ^3.22.0
import { memoize } from 'lodash'; // ^4.17.21

import { 
    IUser, 
    IUserAuth, 
    UserRoleType, 
    UserStatusType 
} from '../interfaces/user.interface';
import { 
    IFleet, 
    IFleetMember, 
    FleetStatus 
} from '../interfaces/fleet.interface';
import { 
    IWebGameState, 
    GameStates, 
    RenderQuality 
} from '../interfaces/game.interface';

// Cache for compiled schemas
const schemaCache = new Map<string, z.ZodSchema>();

/**
 * Enhanced user schema with strict validation rules
 */
const userSchema = z.object({
    id: z.string().uuid(),
    username: z.string()
        .min(3)
        .max(32)
        .regex(/^[a-zA-Z0-9_-]+$/, 'Username must be alphanumeric'),
    email: z.string()
        .email()
        .transform(email => email.toLowerCase()),
    role: z.nativeEnum(UserRoleType),
    deviceCapabilities: z.object({
        lidarSupported: z.boolean(),
        meshNetworkSupported: z.boolean(),
        vulkanVersion: z.string(),
        hardwareSecurityLevel: z.string(),
        scanningResolution: z.number().positive(),
        maxFleetSize: z.number().min(1).max(32)
    }),
    lastActive: z.date(),
    securityLevel: z.string()
}).strict();

/**
 * Enhanced fleet schema with member validation
 */
const fleetSchema = z.object({
    id: z.string().uuid(),
    name: z.string().min(1).max(64),
    maxDevices: z.number().min(1).max(32),
    members: z.array(z.object({
        id: z.string().uuid(),
        deviceId: z.string(),
        role: z.string(),
        connection: z.object({
            lastPing: z.number(),
            connectionQuality: z.number().min(0).max(1),
            retryCount: z.number().min(0)
        }),
        latency: z.number().min(0).max(50),
        connectionQuality: z.object({
            signalStrength: z.number().min(0).max(1),
            stability: z.number().min(0).max(1),
            reliability: z.number().min(0).max(1)
        })
    })).min(1).max(32),
    status: z.nativeEnum(FleetStatus),
    networkStats: z.object({
        averageLatency: z.number().min(0),
        maxLatency: z.number().min(0),
        minLatency: z.number().min(0),
        packetsLost: z.number().min(0),
        bandwidth: z.number().min(0),
        connectedPeers: z.number().min(0).max(32)
    })
}).strict();

/**
 * Enhanced game state schema with performance metrics
 */
const gameStateSchema = z.object({
    gameId: z.string().uuid(),
    sessionId: z.string().uuid(),
    state: z.nativeEnum(GameStates),
    environmentData: z.object({
        meshData: z.instanceof(ArrayBuffer),
        pointCloud: z.instanceof(Float32Array),
        classifiedObjects: z.array(z.object({
            id: z.string(),
            type: z.string(),
            position: z.object({
                x: z.number(),
                y: z.number(),
                z: z.number()
            })
        })),
        timestamp: z.number()
    }).nullable(),
    renderState: z.object({
        resolution: z.object({
            width: z.number().positive(),
            height: z.number().positive()
        }),
        quality: z.nativeEnum(RenderQuality),
        lidarOverlayEnabled: z.boolean()
    }),
    fps: z.number().min(0).max(144)
}).strict();

/**
 * Memoized schema compilation for performance
 */
const getCompiledSchema = memoize((schemaKey: string): z.ZodSchema => {
    if (!schemaCache.has(schemaKey)) {
        switch (schemaKey) {
            case 'user':
                schemaCache.set(schemaKey, userSchema);
                break;
            case 'fleet':
                schemaCache.set(schemaKey, fleetSchema);
                break;
            case 'gameState':
                schemaCache.set(schemaKey, gameStateSchema);
                break;
            default:
                throw new Error(`Unknown schema key: ${schemaKey}`);
        }
    }
    return schemaCache.get(schemaKey)!;
});

/**
 * Sanitizes input string to prevent XSS attacks
 */
const sanitizeInput = (input: string): string => {
    return input.replace(/[<>]/g, '');
};

/**
 * Validates user data with enhanced security checks
 * @throws {z.ZodError} Validation error with details
 */
export const validateUser = (userData: Partial<IUser>): boolean => {
    // Sanitize string inputs
    const sanitizedData = {
        ...userData,
        username: userData.username ? sanitizeInput(userData.username) : undefined,
        email: userData.email ? sanitizeInput(userData.email) : undefined
    };

    // Validate against schema
    const schema = getCompiledSchema('user');
    schema.parse(sanitizedData);

    // Additional security checks
    if (sanitizedData.role === UserRoleType.ADMIN) {
        if (sanitizedData.securityLevel !== 'HIGH') {
            throw new Error('Admin users must have HIGH security level');
        }
    }

    return true;
};

/**
 * Validates fleet data with member status verification
 * @throws {z.ZodError} Validation error with details
 */
export const validateFleet = (fleetData: Partial<IFleet>): boolean => {
    const schema = getCompiledSchema('fleet');
    schema.parse(fleetData);

    // Additional fleet-specific validations
    if (fleetData.members?.length > fleetData.maxDevices!) {
        throw new Error('Fleet size exceeds maxDevices limit');
    }

    // Verify member latencies
    fleetData.members?.forEach(member => {
        if (member.latency > 50) {
            throw new Error(`Member ${member.id} exceeds maximum latency threshold`);
        }
    });

    return true;
};

/**
 * Validates game state with performance optimizations
 * @throws {z.ZodError} Validation error with details
 */
export const validateGameState = (gameState: Partial<IWebGameState>): boolean => {
    const schema = getCompiledSchema('gameState');
    schema.parse(gameState);

    // Validate state transitions
    if (gameState.state === GameStates.RUNNING) {
        if (!gameState.environmentData) {
            throw new Error('Running state requires valid environment data');
        }
        if (gameState.fps! < 30) {
            throw new Error('Running state requires minimum 30 FPS');
        }
    }

    return true;
};

// Export cached schemas for external use
export const schemas = {
    userSchema,
    fleetSchema,
    gameStateSchema
};