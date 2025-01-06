import { z } from 'zod'; // v3.22.2
import { 
    IGameState, IEnvironmentState, IPhysicsState, 
    Vector3, BoundingBox, IClassifiedObject, 
    IPhysicsObject, ICollisionEvent, IFleetMember 
} from '../../interfaces/game.interface';
import { 
    GameStateType, GameConfig, GameCRDTState, FleetConfig 
} from '../../types/game.types';
import { 
    validateGameState, validateFleetState 
} from '../../utils/validation.utils';

// Global validation constants
const MIN_SCAN_QUALITY = 0.6;
const PHYSICS_UPDATE_RATE = 60;
const MAX_CLASSIFIED_OBJECTS = 1000;
const MAX_PHYSICS_OBJECTS = 500;
const MAX_FLEET_SIZE = 32;
const MAX_NETWORK_LATENCY = 50;
const MIN_SCAN_RATE = 30;

// Base vector3 schema for position validation
export const vector3Schema = z.object({
    x: z.number().finite(),
    y: z.number().finite(),
    z: z.number().finite()
}).strict();

// Bounding box schema for spatial validation
const boundingBoxSchema = z.object({
    min: vector3Schema,
    max: vector3Schema
}).refine(
    (box) => box.min.x <= box.max.x && 
             box.min.y <= box.max.y && 
             box.min.z <= box.max.z,
    { message: "Invalid bounding box dimensions" }
);

// Classified object schema for LiDAR detection validation
const classifiedObjectSchema = z.object({
    id: z.string().uuid(),
    type: z.string(),
    position: vector3Schema,
    dimensions: vector3Schema,
    confidence: z.number().min(0).max(1),
    timestamp: z.number().positive()
});

// Physics object schema with collision data validation
const physicsObjectSchema = z.object({
    id: z.string().uuid(),
    position: vector3Schema,
    velocity: vector3Schema,
    mass: z.number().positive(),
    collisionMesh: z.instanceof(ArrayBuffer),
    lastUpdateTimestamp: z.number().positive()
});

// Collision event schema for physics validation
const collisionEventSchema = z.object({
    objectAId: z.string().uuid(),
    objectBId: z.string().uuid(),
    point: vector3Schema,
    force: z.number().nonnegative(),
    timestamp: z.number().positive()
});

// Environment state schema with LiDAR metrics
export const environmentStateSchema = z.object({
    timestamp: z.number().positive(),
    scanQuality: z.number().min(MIN_SCAN_QUALITY).max(1),
    pointCount: z.number().positive(),
    scanLatency: z.number().max(MAX_NETWORK_LATENCY),
    classifiedObjects: z.array(classifiedObjectSchema)
        .max(MAX_CLASSIFIED_OBJECTS)
        .refine(
            (objects) => new Set(objects.map(o => o.id)).size === objects.length,
            { message: "Duplicate object IDs detected" }
        )
});

// Physics state schema with performance validation
export const physicsStateSchema = z.object({
    timestamp: z.number().positive(),
    updateRate: z.number().min(PHYSICS_UPDATE_RATE),
    objects: z.array(physicsObjectSchema)
        .max(MAX_PHYSICS_OBJECTS)
        .refine(
            (objects) => new Set(objects.map(o => o.id)).size === objects.length,
            { message: "Duplicate physics object IDs detected" }
        ),
    collisions: z.array(collisionEventSchema)
});

// Fleet member schema with network metrics
const fleetMemberSchema = z.object({
    id: z.string().uuid(),
    deviceId: z.string(),
    position: vector3Schema,
    latency: z.number().max(MAX_NETWORK_LATENCY),
    lastUpdateTimestamp: z.number().positive()
});

// Fleet state schema with size and performance validation
export const fleetStateSchema = z.object({
    fleetId: z.string().uuid(),
    members: z.array(fleetMemberSchema)
        .max(MAX_FLEET_SIZE)
        .refine(
            (members) => new Set(members.map(m => m.id)).size === members.length,
            { message: "Duplicate member IDs detected" }
        ),
    networkLatency: z.number().max(MAX_NETWORK_LATENCY)
});

// Game state validation function with comprehensive checks
export async function validateGameStateData(
    gameState: IGameState,
    fleetConfig: FleetConfig
): Promise<boolean> {
    try {
        // Validate fleet configuration
        const fleetValidation = fleetStateSchema.safeParse({
            fleetId: gameState.fleetId,
            members: fleetConfig.members,
            networkLatency: gameState.metrics.fleetSyncLatency
        });

        if (!fleetValidation.success) {
            throw new Error(`Fleet validation failed: ${fleetValidation.error.message}`);
        }

        // Validate environment state
        const envValidation = environmentStateSchema.safeParse({
            timestamp: gameState.environment.timestamp,
            scanQuality: gameState.environment.scanQuality,
            pointCount: gameState.environment.pointCount,
            scanLatency: gameState.metrics.lidarProcessingLatency,
            classifiedObjects: gameState.environment.classifiedObjects
        });

        if (!envValidation.success) {
            throw new Error(`Environment validation failed: ${envValidation.error.message}`);
        }

        // Validate physics state
        const physicsValidation = physicsStateSchema.safeParse({
            timestamp: gameState.physics.timestamp,
            updateRate: PHYSICS_UPDATE_RATE,
            objects: gameState.physics.objects,
            collisions: gameState.physics.collisions
        });

        if (!physicsValidation.success) {
            throw new Error(`Physics validation failed: ${physicsValidation.error.message}`);
        }

        // Validate performance metrics
        if (gameState.metrics.stateUpdateLatency > MAX_NETWORK_LATENCY) {
            throw new Error(`State update latency exceeds maximum threshold: ${gameState.metrics.stateUpdateLatency}ms`);
        }

        if (gameState.environment.lidarMetrics.scanRate < MIN_SCAN_RATE) {
            throw new Error(`LiDAR scan rate below minimum threshold: ${gameState.environment.lidarMetrics.scanRate}Hz`);
        }

        // Validate timestamp consistency
        const currentTime = Date.now();
        const maxTimeDrift = 1000; // 1 second maximum drift
        
        if (Math.abs(currentTime - gameState.timestamp) > maxTimeDrift) {
            throw new Error('Game state timestamp exceeds acceptable drift');
        }

        return true;
    } catch (error) {
        console.error('Game state validation failed:', error);
        return false;
    }
}