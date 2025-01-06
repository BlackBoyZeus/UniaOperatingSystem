/**
 * @fileoverview Authentication middleware with hardware-backed security for TALD UNIA platform
 * @version 1.0.0
 * @license MIT
 */

import { Request, Response, NextFunction } from 'express'; // ^4.18.2
import { StatusCodes } from 'http-status'; // ^1.6.2
import * as winston from 'winston'; // ^3.8.2
import { RateLimiterMemory } from 'rate-limiter-flexible'; // ^2.4.1
import { AuthService } from '../../services/auth/AuthService';

// Authentication constants
const AUTH_HEADER = 'Authorization';
const HARDWARE_TOKEN_HEADER = 'X-Hardware-Token';
const TOKEN_TYPE = 'Bearer';
const MAX_AUTH_ATTEMPTS = 5;
const AUTH_WINDOW_MS = 300000; // 5 minutes

// Initialize rate limiter
const rateLimiter = new RateLimiterMemory({
  points: MAX_AUTH_ATTEMPTS,
  duration: AUTH_WINDOW_MS / 1000,
  blockDuration: AUTH_WINDOW_MS / 1000
});

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  defaultMeta: { service: 'auth-middleware' },
  transports: [
    new winston.transports.File({ filename: 'auth-middleware.log' }),
    new winston.transports.Console()
  ]
});

/**
 * Express middleware to authenticate API requests using JWT tokens
 * with hardware token validation and rate limiting
 */
export async function authenticateRequest(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  const ip = req.ip;
  
  try {
    // Check rate limiting
    await rateLimiter.consume(ip);

    // Extract and validate Authorization header
    const authHeader = req.header(AUTH_HEADER);
    if (!authHeader || !authHeader.startsWith(TOKEN_TYPE)) {
      logger.warn('Invalid authorization header', { ip });
      res.status(StatusCodes.UNAUTHORIZED).json({
        error: 'Invalid authorization header'
      });
      return;
    }

    // Extract token
    const token = authHeader.slice(TOKEN_TYPE.length + 1);
    if (!token) {
      logger.warn('Missing access token', { ip });
      res.status(StatusCodes.UNAUTHORIZED).json({
        error: 'Access token required'
      });
      return;
    }

    // Verify token
    const authService = req.app.get('authService') as AuthService;
    const isValid = await authService.verifyAccessToken(token);
    if (!isValid) {
      logger.warn('Invalid access token', { ip });
      res.status(StatusCodes.UNAUTHORIZED).json({
        error: 'Invalid access token'
      });
      return;
    }

    // Token is valid, continue to next middleware
    next();
  } catch (error) {
    if (error.remainingPoints === 0) {
      logger.error('Rate limit exceeded', { ip });
      res.status(StatusCodes.TOO_MANY_REQUESTS).json({
        error: 'Too many authentication attempts',
        retryAfter: error.msBeforeNext
      });
      return;
    }

    logger.error('Authentication error', { error: error.message, ip });
    res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({
      error: 'Authentication failed'
    });
  }
}

/**
 * Express middleware to authenticate device-specific requests
 * using hardware tokens and TPM attestation
 */
export async function authenticateDeviceRequest(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    // Extract hardware token
    const hardwareToken = req.header(HARDWARE_TOKEN_HEADER);
    if (!hardwareToken) {
      logger.warn('Missing hardware token', { ip: req.ip });
      res.status(StatusCodes.UNAUTHORIZED).json({
        error: 'Hardware token required'
      });
      return;
    }

    // Verify hardware token
    const authService = req.app.get('authService') as AuthService;
    const isValid = await authService.authenticateDevice(hardwareToken);
    if (!isValid) {
      logger.warn('Invalid hardware token', { ip: req.ip });
      res.status(StatusCodes.UNAUTHORIZED).json({
        error: 'Invalid hardware token'
      });
      return;
    }

    // Hardware token is valid, continue to next middleware
    next();
  } catch (error) {
    logger.error('Device authentication error', {
      error: error.message,
      ip: req.ip
    });
    res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({
      error: 'Device authentication failed'
    });
  }
}

/**
 * Express middleware to validate fleet-specific access permissions
 * with mesh network validation
 */
export async function validateFleetAccess(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const fleetId = req.params.fleetId || req.body.fleetId;
    if (!fleetId) {
      logger.warn('Missing fleet ID', { ip: req.ip });
      res.status(StatusCodes.BAD_REQUEST).json({
        error: 'Fleet ID required'
      });
      return;
    }

    // Validate fleet membership
    const authService = req.app.get('authService') as AuthService;
    const isValid = await authService.validateFleetMembership(fleetId);
    if (!isValid) {
      logger.warn('Invalid fleet membership', {
        ip: req.ip,
        fleetId
      });
      res.status(StatusCodes.FORBIDDEN).json({
        error: 'Invalid fleet membership'
      });
      return;
    }

    // Fleet access is valid, continue to next middleware
    next();
  } catch (error) {
    logger.error('Fleet validation error', {
      error: error.message,
      ip: req.ip
    });
    res.status(StatusCodes.INTERNAL_SERVER_ERROR).json({
      error: 'Fleet validation failed'
    });
  }
}