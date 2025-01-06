import express, { Router } from 'express'; // v4.18.2
import rateLimit from 'express-rate-limit'; // v6.7.0
import { SessionController } from '../controllers/session.controller';
import { ValidationMiddleware } from '../middlewares/validation.middleware';
import { monitorPerformance } from '@tald/monitoring'; // v1.0.0
import { validateHardwareToken, validateTPM } from '@tald/security-middleware'; // v1.0.0

// Global constants for session management
const SESSION_ROUTES_BASE = '/api/v1/sessions';
const MAX_FLEET_SIZE = 32;
const MAX_LATENCY_MS = 50;
const RATE_LIMIT_WINDOW_MS = 60000;
const RATE_LIMIT_MAX_REQUESTS = 100;

/**
 * Configures session management routes with enhanced security and monitoring
 */
@monitorPerformance()
export function configureSessionRoutes(sessionController: SessionController): Router {
    const router = express.Router();

    // Configure rate limiting for session endpoints
    const sessionRateLimiter = rateLimit({
        windowMs: RATE_LIMIT_WINDOW_MS,
        max: RATE_LIMIT_MAX_REQUESTS,
        message: 'Too many session requests, please try again later'
    });

    // Create session endpoint with fleet validation
    router.post('/',
        sessionRateLimiter,
        validateHardwareToken(),
        validateTPM(),
        ValidationMiddleware.validateRequest(
            'body',
            {
                validateSecurity: true,
                validateFleetContext: true,
                maxLatency: MAX_LATENCY_MS
            }
        ),
        ValidationMiddleware.validateFleetSize(MAX_FLEET_SIZE),
        async (req, res, next) => {
            try {
                const response = await sessionController.createSession(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    // Get session details with security validation
    router.get('/:sessionId',
        sessionRateLimiter,
        validateHardwareToken(),
        ValidationMiddleware.validateRequest(
            'params',
            { validateSecurity: true }
        ),
        async (req, res, next) => {
            try {
                const response = await sessionController.getSession(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    // Update session state with CRDT validation
    router.put('/:sessionId/state',
        sessionRateLimiter,
        validateHardwareToken(),
        ValidationMiddleware.validateRequest(
            'body',
            {
                validateSecurity: true,
                validateCRDT: true,
                validatePerformance: true,
                maxLatency: MAX_LATENCY_MS
            }
        ),
        ValidationMiddleware.validateCRDTState(),
        async (req, res, next) => {
            try {
                const response = await sessionController.updateSessionState(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    // End session with cleanup
    router.delete('/:sessionId',
        sessionRateLimiter,
        validateHardwareToken(),
        ValidationMiddleware.validateRequest(
            'params',
            { validateSecurity: true }
        ),
        async (req, res, next) => {
            try {
                const response = await sessionController.endSession(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    // Check session health with comprehensive monitoring
    router.get('/:sessionId/health',
        sessionRateLimiter,
        validateHardwareToken(),
        ValidationMiddleware.validateRequest(
            'params',
            {
                validateSecurity: true,
                validatePerformance: true
            }
        ),
        async (req, res, next) => {
            try {
                const response = await sessionController.checkSessionHealth(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    // Validate fleet state with performance monitoring
    router.post('/:sessionId/validate',
        sessionRateLimiter,
        validateHardwareToken(),
        ValidationMiddleware.validateRequest(
            'body',
            {
                validateSecurity: true,
                validateFleetContext: true,
                validatePerformance: true,
                maxLatency: MAX_LATENCY_MS
            }
        ),
        ValidationMiddleware.validateFleetState(),
        async (req, res, next) => {
            try {
                const response = await sessionController.validateFleetState(req, res);
                return response;
            } catch (error) {
                next(error);
            }
        }
    );

    return router;
}

export default configureSessionRoutes;