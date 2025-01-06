import { Router } from 'express'; // v4.18.2
import rateLimit from 'express-rate-limit'; // v6.7.0
import { CircuitBreaker } from 'opossum'; // v7.1.0
import correlator from 'express-correlation-id'; // v2.0.0
import { SecurityMiddleware } from '@tald/security-middleware'; // v1.0.0

import { FleetController } from '../controllers/fleet.controller';
import { FleetValidator } from '../validators/fleet.validator';

// Constants for route configuration
const FLEET_BASE_PATH = '/api/v1/fleet';
const RATE_LIMIT_WINDOW = 15 * 60 * 1000; // 15 minutes
const RATE_LIMIT_MAX = 100;
const CIRCUIT_BREAKER_TIMEOUT = 30000;

/**
 * Configures and returns an Express router with enhanced fleet management endpoints
 * Implements comprehensive security, monitoring and WebRTC support
 */
@monitorPerformance
@logRouteAccess
export function configureFleetRoutes(): Router {
    const router = Router();
    const fleetController = new FleetController();
    const fleetValidator = new FleetValidator();

    // Configure middleware
    const limiter = rateLimit({
        windowMs: RATE_LIMIT_WINDOW,
        max: RATE_LIMIT_MAX,
        message: 'Rate limit exceeded for fleet operations'
    });

    const circuitBreaker = new CircuitBreaker(async (req, res, next) => {
        await next();
    }, {
        timeout: CIRCUIT_BREAKER_TIMEOUT,
        errorThresholdPercentage: 50,
        resetTimeout: 30000
    });

    const securityMiddleware = new SecurityMiddleware({
        validateHardwareToken: true,
        enforceEncryption: true,
        trustThreshold: 80
    });

    // Apply global middleware
    router.use(correlator());
    router.use(limiter);
    router.use(securityMiddleware.authenticate());

    // Fleet creation endpoint
    router.post('/',
        fleetValidator.validateFleetCreation,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.createFleet(req, res, next);
        })
    );

    // Fleet join endpoint with hardware token validation
    router.put('/:fleetId/join',
        securityMiddleware.validateHardwareToken,
        fleetValidator.validateFleetUpdate,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.joinFleet(req, res, next);
        })
    );

    // Fleet leave endpoint with state cleanup
    router.put('/:fleetId/leave',
        securityMiddleware.validateMembership,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.leaveFleet(req, res, next);
        })
    );

    // Fleet disband endpoint with cascade deletion
    router.delete('/:fleetId',
        securityMiddleware.validateFleetOwnership,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.disbandFleet(req, res, next);
        })
    );

    // Fleet health monitoring endpoint
    router.get('/:fleetId/health',
        securityMiddleware.validateMembership,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.getFleetHealth(req, res, next);
        })
    );

    // Active fleets listing with pagination
    router.get('/active',
        securityMiddleware.validateAccess,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.getActiveFleets(req, res, next);
        })
    );

    // WebRTC signaling endpoint
    router.post('/:fleetId/signal',
        securityMiddleware.validateMembership,
        fleetValidator.validateWebRTCConfig,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.initiateWebRTCSignaling(req, res, next);
        })
    );

    // Fleet metrics endpoint
    router.get('/:fleetId/metrics',
        securityMiddleware.validateMembership,
        circuitBreaker.fire(async (req, res, next) => {
            await fleetController.getFleetMetrics(req, res, next);
        })
    );

    // Error handling middleware
    router.use((err: Error, req: any, res: any, next: any) => {
        console.error('Fleet route error:', err);
        res.status(500).json({
            success: false,
            error: err.message,
            correlationId: req.correlationId()
        });
    });

    return router;
}

// Export configured router
export const fleetRouter = configureFleetRoutes();