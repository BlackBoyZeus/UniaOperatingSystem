/**
 * @fileoverview Core user interface definitions for TALD UNIA platform
 * @version 1.0.0
 * @license MIT
 */

import { DeviceCapabilities } from '@types/device-capabilities'; // ^1.0.0

/**
 * Core user interface defining required user data fields with enhanced security
 * and device management capabilities. Implements GDPR compliance and hardware-backed
 * authentication support.
 */
export interface IUser {
  /** Unique user identifier */
  id: string;
  
  /** User's chosen username */
  username: string;
  
  /** User's email address (encrypted at rest) */
  email: string;
  
  /** Argon2id password hash */
  passwordHash: string;
  
  /** TPM-backed hardware identifier */
  hardwareId: string;
  
  /** TPM 2.0 public key for hardware attestation */
  tpmPublicKey: string;
  
  /** Device hardware capabilities including LiDAR and mesh networking */
  deviceCapabilities: DeviceCapabilities;
  
  /** User authorization roles */
  roles: string[];
  
  /** GDPR-compliant encrypted fields */
  encryptedFields: Record<string, string>;
  
  /** GDPR consent status */
  gdprConsent: boolean;
  
  /** Last successful login timestamp */
  lastLogin: Date;
  
  /** Account creation timestamp */
  createdAt: Date;
  
  /** Last account update timestamp */
  updatedAt: Date;
  
  /** Scheduled account deletion date (GDPR compliance) */
  deletionScheduled: Date;
}

/**
 * User authentication data interface for token management with 
 * hardware-backed security features.
 */
export interface IUserAuth {
  /** Associated user identifier */
  userId: string;
  
  /** JWT access token */
  accessToken: string;
  
  /** JWT refresh token */
  refreshToken: string;
  
  /** TPM-generated hardware token */
  hardwareToken: string;
  
  /** TPM signature for hardware attestation */
  tpmSignature: string;
  
  /** Device fingerprint for additional security */
  deviceFingerprint: string[];
  
  /** Access token expiration timestamp */
  expiresAt: Date;
  
  /** Refresh token expiration timestamp */
  refreshExpiresAt: Date;
  
  /** Token version for invalidation control */
  tokenVersion: number;
}

/**
 * User profile interface for public-facing user data with configurable
 * privacy controls and preferences.
 */
export interface IUserProfile {
  /** Associated user identifier */
  userId: string;
  
  /** Public display name */
  displayName: string;
  
  /** Avatar URL or identifier */
  avatar: string;
  
  /** Supported languages */
  languages: string[];
  
  /** User preferences key-value store */
  preferences: Record<string, any>;
  
  /** Privacy settings configuration */
  privacySettings: Record<string, boolean>;
  
  /** User timezone */
  timeZone: string;
  
  /** Online status indicator */
  isOnline: boolean;
  
  /** Last known geographic region */
  lastKnownRegion: string;
}