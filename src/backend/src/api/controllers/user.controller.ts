/**
 * @fileoverview User controller with hardware-backed security and fleet management
 * @version 1.0.0
 * @license MIT
 */

import { Request, Response } from 'express';
import * as winston from 'winston';
import { RateLimiterFlexible } from 'rate-limiter-flexible';
import { IUser, IUserAuth, IUserProfile } from '../../interfaces/user.interface';
import { AuthService } from '../../services/auth/AuthService';
import { generateHash, generateHMAC } from '../../utils/crypto.utils';

// Rate limiting configuration
const RATE_LIMIT_CONFIG = {
  points: 5, // Number of attempts
  duration: 300, // 5 minutes
  blockDuration: 900, // 15 minutes block
  hardwareIdPoints: 3 // Hardware-specific limits
};

// GDPR compliance settings
const GDPR_RETENTION_PERIOD = 30 * 24 * 60 * 60 * 1000; // 30 days in milliseconds
const GDPR_REQUIRED_FIELDS = ['email', 'username', 'hardwareId'];

@controller('/api/users')
@useRateLimiter()
@auditLog()
export class UserController {
  private readonly authService: AuthService;
  private readonly logger: winston.Logger;
  private readonly rateLimiter: RateLimiterFlexible;

  constructor(
    authService: AuthService,
    logger: winston.Logger,
    rateLimiter: RateLimiterFlexible
  ) {
    this.authService = authService;
    this.logger = logger;
    this.rateLimiter = rateLimiter;

    // Initialize enhanced logging
    this.logger.configure({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      defaultMeta: { service: 'user-controller' }
    });
  }

  /**
   * Register new user with hardware-backed security
   */
  @post('/register')
  @validate(userSchema)
  @rateLimit('register')
  @auditLog('user-registration')
  public async register(req: Request, res: Response): Promise<Response> {
    try {
      const { username, email, password, hardwareId, tpmPublicKey, deviceCapabilities } = req.body;

      // Validate hardware ID and TPM signature
      const tpmSignature = req.headers['x-tpm-signature'] as string;
      if (!tpmSignature || !await this.authService.verifyTPMSignature(hardwareId, tpmSignature)) {
        throw new Error('Invalid hardware signature');
      }

      // Validate device capabilities
      if (!await this.authService.validateDeviceCapabilities(deviceCapabilities)) {
        throw new Error('Insufficient device capabilities');
      }

      // Apply rate limiting by hardware ID
      await this.rateLimiter.consume(`hw:${hardwareId}`, RATE_LIMIT_CONFIG.hardwareIdPoints);

      // Generate secure password hash
      const { hash: passwordHash, metadata: hashMetadata } = await generateHash(password, {
        iterations: 100000,
        hardwareAcceleration: true
      });

      // Create user with hardware binding
      const user: IUser = {
        id: crypto.randomUUID(),
        username,
        email,
        passwordHash,
        hardwareId,
        tpmPublicKey,
        deviceCapabilities,
        roles: ['user'],
        encryptedFields: {},
        gdprConsent: false,
        lastLogin: new Date(),
        createdAt: new Date(),
        updatedAt: new Date(),
        deletionScheduled: new Date(Date.now() + GDPR_RETENTION_PERIOD)
      };

      // Validate GDPR requirements
      this.validateGDPRCompliance(user);

      // Generate hardware-bound authentication tokens
      const auth: IUserAuth = await this.authService.authenticateDevice(user, hardwareId);

      // Create public profile
      const profile: IUserProfile = {
        userId: user.id,
        displayName: username,
        avatar: '',
        languages: [],
        preferences: {},
        privacySettings: {
          shareOnlineStatus: false,
          shareLocation: false
        },
        timeZone: req.headers['x-timezone'] as string || 'UTC',
        isOnline: true,
        lastKnownRegion: req.headers['x-region'] as string || 'unknown'
      };

      // Log successful registration
      this.logger.info('User registered successfully', {
        userId: user.id,
        hardwareId,
        deviceCapabilities,
        hashMetadata
      });

      return res.status(201).json({
        user: this.sanitizeUserData(user),
        auth,
        profile
      });

    } catch (error) {
      this.logger.error('Registration failed', {
        error: error.message,
        hardwareId: req.body.hardwareId
      });

      return res.status(400).json({
        error: 'Registration failed',
        message: error.message
      });
    }
  }

  /**
   * Sanitize user data for GDPR compliance
   */
  private sanitizeUserData(user: IUser): Partial<IUser> {
    const { passwordHash, encryptedFields, ...sanitizedUser } = user;
    return sanitizedUser;
  }

  /**
   * Validate GDPR compliance requirements
   */
  private validateGDPRCompliance(user: IUser): void {
    for (const field of GDPR_REQUIRED_FIELDS) {
      if (!user[field]) {
        throw new Error(`Missing required GDPR field: ${field}`);
      }
    }
  }
}