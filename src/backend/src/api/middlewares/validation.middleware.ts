import { Request, Response, NextFunction } from 'express'; // v4.18.2
import { z } from 'zod'; // v3.22.2
import now from 'performance-now'; // v2.1.0

import { 
    validateFleetConfiguration, 
    validateGameState, 
    validateLidarConfig, 
    validateSecurityContext 
} from '../../utils/validation.utils';

import {
    createFleetSchema,
    joinFleetSchema,
    updateFleetStateSchema,
    fleetTopologySchema
} from '../validators/fleet.validator';

import { TaldError } from '../middlewares/error.middleware';

// Global validation constants
const MAX_FLEET_SIZE = 32;
const MAX_LATENCY = 50;
const MIN_SCAN_RESOLUTION = 0.01;
const MAX_SCAN_RANGE = 5.0;
const VALIDATION_TIMEOUT = 100;
const MAX_BATCH_SIZE = 1000;

/**
 * Enhanced request validation options interface
 */
interface ValidationOptions {
    validateSecurity?: boolean;
    validatePerformance?: boolean;
    validateFleetContext?: boolean;
    validateCRDT?: boolean;
    maxLatency?: number;
    timeout?: number;
}

/**
 * Enhanced validation metrics interface
 */
interface ValidationMetrics {
    validationTime: number;
    validationStart: number;
    validationEnd: number;
    schemaValidationTime: number;
    securityValidationTime?: number;
    crdtValidationTime?: number;
}

/**
 * Generic request validation middleware factory with enhanced security and performance monitoring
 */
export function validateRequest(
    schema: z.ZodSchema,
    location: 'body' | 'params' | 'query',
    options: ValidationOptions = {}
) {
    return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
        const metrics: ValidationMetrics = {
            validationTime: 0,
            validationStart: now(),
            validationEnd: 0,
            schemaValidationTime: 0
        };

        try {
            // Start validation timing
            const schemaValidationStart = now();

            // Extract data from request
            const data = req[location];

            // Validate security context if enabled
            if (options.validateSecurity) {
                const securityStart = now();
                await validateSecurityContext(req);
                metrics.securityValidationTime = now() - securityStart;
            }

            // Validate schema
            const validationResult = await schema.safeParseAsync(data);

            metrics.schemaValidationTime = now() - schemaValidationStart;

            if (!validationResult.success) {
                throw new TaldError(
                    'Validation failed',
                    400,
                    { errors: validationResult.error.errors },
                    { validationMetrics: metrics }
                );
            }

            // Attach validated data to request
            req[location] = validationResult.data;

            // Complete validation timing
            metrics.validationEnd = now();
            metrics.validationTime = metrics.validationEnd - metrics.validationStart;

            // Check validation performance
            if (options.validatePerformance && metrics.validationTime > (options.timeout || VALIDATION_TIMEOUT)) {
                throw new TaldError(
                    'Validation timeout exceeded',
                    408,
                    { validationMetrics: metrics }
                );
            }

            next();
        } catch (error) {
            next(error);
        }
    };
}

/**
 * Enhanced fleet request validation middleware with topology and CRDT verification
 */
export async function validateFleetRequest(
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> {
    const validationStart = now();

    try {
        const fleetData = req.body;
        const fleetId = req.params.fleetId;

        // Validate fleet configuration
        const fleetValidation = await validateFleetConfiguration(
            fleetData,
            req.headers['x-user-role'] as string,
            fleetData.meshConfig
        );

        if (!fleetValidation.valid) {
            throw new TaldError(
                'Fleet validation failed',
                400,
                { errors: fleetValidation.errors }
            ).withFleetContext({ fleetId });
        }

        // Validate mesh topology
        const topologyValidation = await fleetTopologySchema.safeParseAsync(fleetData.meshConfig.topology);
        if (!topologyValidation.success) {
            throw new TaldError(
                'Invalid mesh topology configuration',
                400,
                { errors: topologyValidation.error.errors }
            ).withFleetContext({ fleetId });
        }

        // Track validation performance
        const validationTime = now() - validationStart;
        if (validationTime > VALIDATION_TIMEOUT) {
            throw new TaldError(
                'Fleet validation timeout exceeded',
                408,
                { validationTime }
            ).withFleetContext({ fleetId });
        }

        next();
    } catch (error) {
        next(error);
    }
}

/**
 * Enhanced game state validation middleware with CRDT verification
 */
export async function validateGameStateRequest(
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> {
    const validationStart = now();

    try {
        const gameState = req.body;
        const fleetId = req.headers['x-fleet-id'] as string;

        // Validate game state
        const stateValidation = await validateGameState(
            gameState,
            gameState.environment,
            gameState.physics
        );

        if (!stateValidation.valid) {
            throw new TaldError(
                'Game state validation failed',
                400,
                { metrics: stateValidation.metrics }
            ).withFleetContext({ fleetId });
        }

        // Track validation performance
        const validationTime = now() - validationStart;
        if (validationTime > VALIDATION_TIMEOUT) {
            throw new TaldError(
                'Game state validation timeout exceeded',
                408,
                { validationTime }
            ).withFleetContext({ fleetId });
        }

        next();
    } catch (error) {
        next(error);
    }
}

/**
 * Enhanced LiDAR data validation middleware with performance monitoring
 */
export async function validateLidarRequest(
    req: Request,
    res: Response,
    next: NextFunction
): Promise<void> {
    const validationStart = now();

    try {
        const lidarConfig = req.body;
        const scanMetadata = req.body.metadata;
        const calibrationData = req.body.calibration;

        // Validate LiDAR configuration
        const lidarValidation = await validateLidarConfig(
            lidarConfig,
            calibrationData,
            scanMetadata
        );

        if (!lidarValidation.valid) {
            throw new TaldError(
                'LiDAR validation failed',
                400,
                { quality: lidarValidation.quality }
            );
        }

        // Validate scan resolution
        if (lidarConfig.resolution < MIN_SCAN_RESOLUTION) {
            throw new TaldError(
                'Scan resolution below minimum threshold',
                400,
                { minResolution: MIN_SCAN_RESOLUTION }
            );
        }

        // Validate scan range
        if (lidarConfig.range > MAX_SCAN_RANGE) {
            throw new TaldError(
                'Scan range exceeds maximum limit',
                400,
                { maxRange: MAX_SCAN_RANGE }
            );
        }

        // Track validation performance
        const validationTime = now() - validationStart;
        if (validationTime > VALIDATION_TIMEOUT) {
            throw new TaldError(
                'LiDAR validation timeout exceeded',
                408,
                { validationTime }
            );
        }

        next();
    } catch (error) {
        next(error);
    }
}