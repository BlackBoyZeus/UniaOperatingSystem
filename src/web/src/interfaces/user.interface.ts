// External imports
import { UUID } from 'crypto'; // v18.0.0+ - Cryptographically secure UUID type

/**
 * Defines user role types for RBAC implementation
 */
export enum UserRoleType {
    USER = 'USER',
    FLEET_LEADER = 'FLEET_LEADER',
    DEVELOPER = 'DEVELOPER',
    ADMIN = 'ADMIN'
}

/**
 * Comprehensive device hardware and capability types
 */
export interface DeviceCapabilityType {
    lidarSupported: boolean;
    meshNetworkSupported: boolean;
    vulkanVersion: string;
    hardwareSecurityLevel: string;
    scanningResolution: number;
    maxFleetSize: number;
}

/**
 * Extended user status type definitions for fleet management
 */
export enum UserStatusType {
    ONLINE = 'ONLINE',
    OFFLINE = 'OFFLINE',
    IN_GAME = 'IN_GAME',
    IN_FLEET = 'IN_FLEET',
    SCANNING = 'SCANNING'
}

/**
 * GDPR-compliant privacy settings interface
 */
export interface PrivacySettingsType {
    shareLocation: boolean;
    shareScanData: boolean;
    dataRetentionDays: number;
    gdprConsent: boolean;
}

/**
 * Extended user preferences with privacy and fleet settings
 */
export interface UserPreferencesType {
    theme: string;
    notifications: boolean;
    privacy: PrivacySettingsType;
    fleetPreferences: {
        autoJoin: boolean;
        preferredRole: string;
        maxFleetSize: number;
    };
    scanningPreferences: {
        resolution: number;
        autoShare: boolean;
        retentionPolicy: string;
    };
}

/**
 * Core user interface with enhanced security features
 */
export interface IUser {
    id: UUID;
    username: string;
    email: string;
    role: UserRoleType;
    deviceCapabilities: DeviceCapabilityType;
    lastActive: Date;
    securityLevel: string;
}

/**
 * Enhanced authentication interface with hardware security
 */
export interface IUserAuth {
    userId: UUID;
    accessToken: string;
    refreshToken: string;
    hardwareToken: string;
    expiresAt: Date;
    tpmSignature: string;
}

/**
 * Extended user profile with fleet and scanning status
 */
export interface IUserProfile {
    userId: UUID;
    displayName: string;
    avatar: string;
    status: UserStatusType;
    preferences: UserPreferencesType;
    fleetId: UUID;
    activeScans: number;
}