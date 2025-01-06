/**
 * @fileoverview Express router configuration for user-related endpoints with hardware-backed security
 * @version 1.0.0
 * @license MIT
 */

import { Router } from 'express'; // v4.18.2
import rateLimit from 'express-rate-limit'; // v6.7.0
import winston from 'winston'; // v3.8.2
import { UserController } from '../controllers/user.controller';
import { authenticateRequest, authenticateDeviceRequest, validateFleetAccess } from '../middlewares/auth.middleware';
import { validateRequest } from '../middlewares/validation.middleware';
import { TaldLogger } from '../../utils/logger.utils';

// Initialize enhanced logger
const logger = new TaldLogger({
    serviceName: 'UserRoutes',
    environment: process.env.NODE_ENV || 'development',
    enableCloudWatch: true,
    securitySettings: {
        trackAuthEvents: true,
        trackSystemIntegrity: true,
        fleetTrustThreshold: 80
    },
    privacySettings: {
        maskPII: true,
        sensitiveFields: ['password', 'token', 'hardwareId', 'tpmPublicKey']
    },
    performanceTracking: true
});

// Rate limiting configurations
const registrationLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts per window
    message: 'Too many registration attempts, please try again later'
});

const loginLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 10, // 10 attempts per window
    message: 'Too many login attempts, please try again later'
});

const profileLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 30, // 30 requests per window
    message: 'Too many profile requests, please try again later'
});

/**
 * Initializes user routes with enhanced security features
 * @param userController Initialized UserController instance
 * @returns Configured Express router
 */
export function initializeUserRoutes(userController: UserController): Router {
    const router = Router();

    // Registration endpoint with hardware validation
    router.post('/register',
        registrationLimiter,
        validateRequest({
            body: ['username', 'email', 'password', 'hardwareId', 'tpmPublicKey', 'deviceCapabilities']
        }),
        userController.validateHardwareToken,
        async (req, res, next) => {
            try {
                const result = await userController.register(req, res);
                logger.info('User registered successfully', {
                    userId: result.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Registration failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Login endpoint with hardware token verification
    router.post('/login',
        loginLimiter,
        validateRequest({
            body: ['username', 'password', 'hardwareId', 'tpmSignature']
        }),
        userController.validateHardwareToken,
        async (req, res, next) => {
            try {
                const result = await userController.login(req, res);
                logger.info('User logged in successfully', {
                    userId: result.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Login failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Token refresh endpoint with hardware re-validation
    router.post('/refresh',
        rateLimit({
            windowMs: 15 * 60 * 1000,
            max: 20
        }),
        authenticateRequest,
        userController.validateHardwareToken,
        async (req, res, next) => {
            try {
                const result = await userController.refreshToken(req, res);
                logger.debug('Token refreshed', {
                    userId: req.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Token refresh failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Logout endpoint with token revocation
    router.post('/logout',
        rateLimit({
            windowMs: 5 * 60 * 1000,
            max: 10
        }),
        authenticateRequest,
        async (req, res, next) => {
            try {
                const result = await userController.logout(req, res);
                logger.info('User logged out', {
                    userId: req.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Logout failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Profile retrieval endpoint with security validation
    router.get('/profile',
        profileLimiter,
        authenticateRequest,
        async (req, res, next) => {
            try {
                const result = await userController.getProfile(req, res);
                logger.debug('Profile retrieved', {
                    userId: req.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Profile retrieval failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Profile update endpoint with security validation
    router.put('/profile',
        rateLimit({
            windowMs: 5 * 60 * 1000,
            max: 10
        }),
        authenticateRequest,
        validateRequest({
            body: ['displayName', 'avatar', 'preferences', 'privacySettings']
        }),
        async (req, res, next) => {
            try {
                const result = await userController.updateProfile(req, res);
                logger.info('Profile updated', {
                    userId: req.user.id,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Profile update failed', {
                    error: error.message,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    // Fleet-specific profile endpoint with mesh validation
    router.get('/profile/fleet/:fleetId',
        profileLimiter,
        authenticateRequest,
        authenticateDeviceRequest,
        validateFleetAccess,
        async (req, res, next) => {
            try {
                const result = await userController.getFleetProfile(req, res);
                logger.debug('Fleet profile retrieved', {
                    userId: req.user.id,
                    fleetId: req.params.fleetId,
                    deviceId: req.body.hardwareId
                });
                return result;
            } catch (error) {
                logger.error('Fleet profile retrieval failed', {
                    error: error.message,
                    fleetId: req.params.fleetId,
                    deviceId: req.body.hardwareId
                });
                next(error);
            }
        }
    );

    return router;
}

export const userRouter = initializeUserRoutes(new UserController());