/**
 * @fileoverview Enhanced user data validation with hardware security and GDPR compliance
 * @version 1.0.0
 * @license MIT
 */

import { z } from 'zod'; // ^3.22.2
import { isEmail, isStrongPassword } from 'validator'; // ^13.11.0
import { SecurityUtils } from '@tald/security-utils'; // ^1.0.0
import { IUser, IUserAuth, IUserProfile } from '../../interfaces/user.interface';

// Global validation constants
const PASSWORD_MIN_LENGTH = 16;
const PASSWORD_PATTERN = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{16,}$/;
const HARDWARE_ID_PATTERN = /^TALD-[A-F0-9]{32}$/;
const TPM_SIGNATURE_PATTERN = /^TPM-[A-F0-9]{64}$/;

/**
 * Enhanced device capabilities validation schema
 */
const deviceCapabilitiesSchema = z.object({
  lidar: z.object({
    supported: z.boolean(),
    resolution: z.number().min(0.01),
    scanRate: z.number().min(30),
    range: z.number().min(5)
  }),
  network: z.object({
    meshSupported: z.boolean(),
    maxPeers: z.number().min(32),
    latency: z.number().max(50)
  }),
  security: z.object({
    tpmVersion: z.string().min(4),
    secureEnclave: z.boolean(),
    biometrics: z.boolean()
  }),
  hardware: z.object({
    processor: z.string(),
    memory: z.number().min(4096),
    storage: z.number().min(32768)
  })
});

/**
 * Enhanced encryption keys schema with TPM integration
 */
const encryptionKeysSchema = z.object({
  publicKey: z.string().min(32),
  keyId: z.string().uuid(),
  algorithm: z.enum(['AES-256-GCM', 'ChaCha20-Poly1305']),
  createdAt: z.date(),
  rotationDue: z.date()
});

/**
 * Enhanced user data validation schema with hardware security
 */
export const userSchema = z.object({
  id: z.string().uuid(),
  username: z.string().min(3).max(30),
  email: z.string().email(),
  passwordHash: z.string().min(97).max(97), // Argon2id hash length
  hardwareId: z.string().regex(HARDWARE_ID_PATTERN),
  tpmPublicKey: z.string().regex(TPM_SIGNATURE_PATTERN),
  deviceCapabilities: deviceCapabilitiesSchema,
  roles: z.array(z.string()),
  encryptionKeys: encryptionKeysSchema,
  gdprConsent: z.boolean(),
  lastLogin: z.date(),
  createdAt: z.date(),
  updatedAt: z.date(),
  deletionScheduled: z.date().optional()
});

/**
 * Enhanced authentication validation schema with TPM integration
 */
export const userAuthSchema = z.object({
  userId: z.string().uuid(),
  accessToken: z.string().min(128),
  refreshToken: z.string().min(128),
  hardwareToken: z.string().min(64),
  tpmSignature: z.string().regex(TPM_SIGNATURE_PATTERN),
  deviceFingerprint: z.array(z.string()),
  expiresAt: z.date(),
  refreshExpiresAt: z.date(),
  tokenVersion: z.number().min(0)
});

/**
 * Enhanced user profile validation schema with privacy controls
 */
export const userProfileSchema = z.object({
  userId: z.string().uuid(),
  displayName: z.string().min(3).max(30),
  avatar: z.string().url().optional(),
  languages: z.array(z.string()),
  preferences: z.record(z.string(), z.any()),
  privacySettings: z.record(z.string(), z.boolean()),
  timeZone: z.string(),
  isOnline: z.boolean(),
  lastKnownRegion: z.string()
});

/**
 * Enhanced user data validation with hardware security checks
 * @param userData - User data to validate
 * @param securityUtils - Security utilities instance
 * @returns Promise<boolean> - Returns true if validation passes, throws error if invalid
 */
export async function validateUser(
  userData: Record<string, any>,
  securityUtils: SecurityUtils
): Promise<boolean> {
  try {
    // Validate basic user schema
    const validatedUser = userSchema.parse(userData);

    // Enhanced email validation
    if (!isEmail(validatedUser.email)) {
      throw new Error('Invalid email format');
    }

    // Enhanced password validation
    if (!isStrongPassword(userData.password, {
      minLength: PASSWORD_MIN_LENGTH,
      minLowercase: 1,
      minUppercase: 1,
      minNumbers: 1,
      minSymbols: 1
    })) {
      throw new Error('Password does not meet security requirements');
    }

    // Verify TPM and hardware security
    const isValidTPM = await securityUtils.verifyTPMSignature(
      validatedUser.hardwareId,
      validatedUser.tpmPublicKey
    );
    if (!isValidTPM) {
      throw new Error('Invalid TPM signature');
    }

    // Verify device capabilities
    const hasRequiredCapabilities = await securityUtils.verifyDeviceCapabilities(
      validatedUser.deviceCapabilities
    );
    if (!hasRequiredCapabilities) {
      throw new Error('Device does not meet minimum requirements');
    }

    // Verify encryption configuration
    const isValidEncryption = await securityUtils.verifyEncryptionConfig(
      validatedUser.encryptionKeys
    );
    if (!isValidEncryption) {
      throw new Error('Invalid encryption configuration');
    }

    // GDPR compliance check
    if (!validatedUser.gdprConsent) {
      throw new Error('GDPR consent required');
    }

    return true;
  } catch (error) {
    throw new Error(`Validation failed: ${error.message}`);
  }
}