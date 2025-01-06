import { Router } from 'express'; // ^4.18.2
import { rateLimit } from 'express-rate-limit'; // ^6.7.0
import { CircuitBreaker } from 'opossum'; // ^7.1.0
import { PerformanceMonitor } from '@performance-monitor/core'; // ^2.0.0

import { GameController } from '../controllers/game.controller';
import {
    authenticateRequest,
    authenticateDeviceRequest,
    validateFleetAccess
} from '../middlewares/auth.middleware';
import { validateRequest } from '../middlewares/validation.middleware';
import {
    gameStateSchema,
    environmentStateSchema
} from '../validators/game.validator';
import { ErrorHandler } from '../middlewares/error.middleware';

// Global constants for rate limiting and performance thresholds
const GAME_ROUTES_BASE_PATH = '/api/v1/games';
const RATE_LIMIT_WINDOWS = {
    CREATE_SESSION: '10/minute',
    JOIN_SESSION: '20/minute',
    UPDATE_ENVIRONMENT: '30/second',
    END_SESSION: '5/minute'
};
const PERFORMANCE_THRESHOLDS = {
    MAX_LATENCY: 50, // ms
    BATCH_SIZE: 100,
    SYNC_INTERVAL: 16 // ms
};

// Circuit breaker configuration
const CIRCUIT_BREAKER_OPTIONS = {
    timeout: 3000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
};

/**
 * Configures and returns game-related routes with enhanced security,
 * performance monitoring, and fleet-aware error handling
 */
export function configureGameRoutes(
    gameController: GameController,
    performanceMonitor: PerformanceMonitor,
    errorHandler: ErrorHandler
): Router {
    const router = Router();

    // Initialize circuit breakers
    const sessionBreaker = new CircuitBreaker(gameController.createSession, CIRCUIT_BREAKER_OPTIONS);
    const environmentBreaker = new CircuitBreaker(gameController.updateEnvironment, CIRCUIT_BREAKER_OPTIONS);

    // Rate limiters for different endpoints
    const createSessionLimiter = rateLimit({
        windowMs: 60 * 1000,
        max: 10,
        message: 'Too many session creation attempts'
    });

    const joinSessionLimiter = rateLimit({
        windowMs: 60 * 1000,
        max: 20,
        message: 'Too many session join attempts'
    });

    const updateEnvironmentLimiter = rateLimit({
        windowMs: 1000,
        max: 30,
        message: 'Environment update rate exceeded'
    });

    // Create new game session
    router.post('/sessions',
        authenticateDeviceRequest,
        createSessionLimiter,
        validateRequest(gameStateSchema, 'body', {
            validateSecurity: true,
            validatePerformance: true
        }),
        async (req, res, next) => {
            try {
                const startTime = performance.now();
                const session = await sessionBreaker.fire(req.body);

                performanceMonitor.recordMetric('session_creation_latency', performance.now() - startTime);

                res.status(201).json(session);
            } catch (error) {
                next(error);
            }
        }
    );

    // Join existing game session
    router.post('/sessions/:gameId/join',
        authenticateDeviceRequest,
        validateFleetAccess,
        joinSessionLimiter,
        async (req, res, next) => {
            try {
                const startTime = performance.now();
                await gameController.joinSession(req.params.gameId, req.body);

                performanceMonitor.recordMetric('session_join_latency', performance.now() - startTime);

                res.status(200).json({ success: true });
            } catch (error) {
                next(error);
            }
        }
    );

    // Update environment state with batch processing
    router.put('/sessions/:gameId/environment',
        authenticateDeviceRequest,
        validateFleetAccess,
        updateEnvironmentLimiter,
        validateRequest(environmentStateSchema, 'body', {
            validatePerformance: true
        }),
        async (req, res, next) => {
            try {
                const startTime = performance.now();
                const updates = Array.isArray(req.body) ? req.body : [req.body];

                if (updates.length > PERFORMANCE_THRESHOLDS.BATCH_SIZE) {
                    throw new Error(`Batch size exceeds maximum of ${PERFORMANCE_THRESHOLDS.BATCH_SIZE}`);
                }

                await environmentBreaker.fire(req.params.gameId, updates);

                const latency = performance.now() - startTime;
                performanceMonitor.recordMetric('environment_update_latency', latency);

                if (latency > PERFORMANCE_THRESHOLDS.MAX_LATENCY) {
                    performanceMonitor.recordEvent('high_latency_warning', {
                        latency,
                        gameId: req.params.gameId
                    });
                }

                res.status(200).json({
                    success: true,
                    latency,
                    batchSize: updates.length
                });
            } catch (error) {
                next(error);
            }
        }
    );

    // End game session with cleanup
    router.delete('/sessions/:gameId',
        authenticateDeviceRequest,
        validateFleetAccess,
        rateLimit({
            windowMs: 60 * 1000,
            max: 5,
            message: 'Too many session termination attempts'
        }),
        async (req, res, next) => {
            try {
                const startTime = performance.now();
                await gameController.endSession(req.params.gameId);

                performanceMonitor.recordMetric('session_cleanup_latency', performance.now() - startTime);

                res.status(200).json({ success: true });
            } catch (error) {
                next(error);
            }
        }
    );

    // Get session performance metrics
    router.get('/sessions/:gameId/metrics',
        authenticateDeviceRequest,
        validateFleetAccess,
        async (req, res, next) => {
            try {
                const metrics = await gameController.getSessionMetrics(req.params.gameId);
                res.status(200).json(metrics);
            } catch (error) {
                next(error);
            }
        }
    );

    // Error handling middleware
    router.use(errorHandler);

    return router;
}