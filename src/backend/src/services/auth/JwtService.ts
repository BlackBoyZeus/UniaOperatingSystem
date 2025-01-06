/**
 * @fileoverview Enhanced JWT service implementation with hardware-backed security
 * @version 1.0.0
 * @license MIT
 */

import { injectable } from 'inversify';
import * as jwt from 'jsonwebtoken'; // ^9.0.0
import * as AWS from 'aws-sdk'; // ^2.1400.0
import * as winston from 'winston'; // ^3.8.0
import { RateLimiterMemory } from 'rate-limiter-flexible'; // ^2.4.1
import { IUser } from '../../interfaces/user.interface';

// Token expiry times in seconds
const ACCESS_TOKEN_EXPIRY = 3600; // 1 hour
const REFRESH_TOKEN_EXPIRY = 604800; // 1 week
const HARDWARE_TOKEN_EXPIRY = 31536000; // 1 year
const MAX_TOKEN_ATTEMPTS = 5;
const RATE_LIMIT_WINDOW = 300; // 5 minutes

@injectable()
export class JwtService {
  private kmsClient: AWS.KMS;
  private signingKey: string;
  private logger: winston.Logger;
  private rateLimiter: RateLimiterMemory;
  private revokedTokens: Map<string, Set<string>>;

  constructor(
    awsConfig: AWS.Config,
    logger: winston.Logger,
    rateLimiter: RateLimiterMemory
  ) {
    this.kmsClient = new AWS.KMS(awsConfig);
    this.logger = logger;
    this.rateLimiter = rateLimiter;
    this.revokedTokens = new Map();
    this.initializeService();
  }

  /**
   * Initialize service with hardware security module integration
   */
  private async initializeService(): Promise<void> {
    try {
      const keyResponse = await this.kmsClient.createKey({
        KeyUsage: 'SIGN_VERIFY',
        CustomerMasterKeySpec: 'ECC_NIST_P256',
        Description: 'TALD UNIA JWT Signing Key'
      }).promise();

      this.signingKey = keyResponse.KeyMetadata.KeyId;
      this.logger.info('JWT service initialized with hardware security module', {
        keyId: this.signingKey
      });
    } catch (error) {
      this.logger.error('Failed to initialize JWT service', { error });
      throw new Error('JWT service initialization failed');
    }
  }

  /**
   * Generate hardware-backed JWT access token
   */
  public async generateAccessToken(user: IUser): Promise<string> {
    try {
      // Apply rate limiting
      await this.rateLimiter.consume(user.id);

      // Validate hardware binding
      if (!user.hardwareId) {
        throw new Error('Hardware ID required for token generation');
      }

      const payload = {
        uid: user.id,
        hwid: user.hardwareId,
        type: 'access'
      };

      // Sign token with hardware security module
      const signParams = {
        KeyId: this.signingKey,
        Message: Buffer.from(JSON.stringify(payload)),
        SigningAlgorithm: 'ECDSA_SHA_256'
      };

      const signature = await this.kmsClient.sign(signParams).promise();
      
      const token = jwt.sign(payload, signature.Signature.toString('base64'), {
        expiresIn: ACCESS_TOKEN_EXPIRY,
        algorithm: 'ES256'
      });

      this.logger.info('Access token generated', {
        userId: user.id,
        tokenId: jwt.decode(token)['jti']
      });

      return token;
    } catch (error) {
      this.logger.error('Token generation failed', {
        userId: user.id,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Verify token with enhanced security checks
   */
  public async verifyToken(token: string): Promise<boolean> {
    try {
      // Check revocation status
      const decoded = jwt.decode(token) as jwt.JwtPayload;
      if (!decoded || this.isTokenRevoked(decoded.uid, token)) {
        return false;
      }

      // Verify signature with hardware security module
      const verifyParams = {
        KeyId: this.signingKey,
        Message: Buffer.from(JSON.stringify({
          uid: decoded.uid,
          hwid: decoded.hwid,
          type: decoded.type
        })),
        Signature: Buffer.from(decoded.signature, 'base64'),
        SigningAlgorithm: 'ECDSA_SHA_256'
      };

      await this.kmsClient.verify(verifyParams).promise();

      this.logger.debug('Token verified successfully', {
        tokenId: decoded.jti
      });

      return true;
    } catch (error) {
      this.logger.warn('Token verification failed', {
        error: error.message
      });
      return false;
    }
  }

  /**
   * Revoke specific token or all user tokens
   */
  public async revokeToken(userId: string, token?: string): Promise<void> {
    try {
      if (!this.revokedTokens.has(userId)) {
        this.revokedTokens.set(userId, new Set());
      }

      if (token) {
        this.revokedTokens.get(userId).add(token);
        this.logger.info('Token revoked', {
          userId,
          tokenId: jwt.decode(token)['jti']
        });
      } else {
        // Revoke all tokens for user
        const userTokens = this.revokedTokens.get(userId);
        userTokens.clear();
        this.logger.info('All tokens revoked for user', { userId });
      }
    } catch (error) {
      this.logger.error('Token revocation failed', {
        userId,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Check if token is revoked
   */
  private isTokenRevoked(userId: string, token: string): boolean {
    const userTokens = this.revokedTokens.get(userId);
    return userTokens ? userTokens.has(token) : false;
  }

  /**
   * Clean up expired revoked tokens
   */
  private async cleanupRevokedTokens(): Promise<void> {
    for (const [userId, tokens] of this.revokedTokens.entries()) {
      for (const token of tokens) {
        try {
          const decoded = jwt.decode(token) as jwt.JwtPayload;
          if (decoded.exp * 1000 < Date.now()) {
            tokens.delete(token);
          }
        } catch (error) {
          tokens.delete(token);
        }
      }
      if (tokens.size === 0) {
        this.revokedTokens.delete(userId);
      }
    }
  }
}