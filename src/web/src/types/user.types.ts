// External imports
import { UUID } from 'crypto'; // v18.0.0+

// Internal imports
import { IUser } from '../interfaces/user.interface';

/**
 * Comprehensive user role type definition with granular permissions
 */
export type UserRole = {
    readonly USER: 'USER';
    readonly FLEET_LEADER: 'FLEET_LEADER';
    readonly DEVELOPER: 'DEVELOPER';
    readonly ADMIN: 'ADMIN';
};

/**
 * Extended user status including fleet and game states
 */
export type UserStatus = {
    readonly ONLINE: 'ONLINE';
    readonly OFFLINE: 'OFFLINE';
    readonly IN_GAME: 'IN_GAME';
    readonly IN_FLEET: 'IN_FLEET';
};

/**
 * Comprehensive device capability type definition
 * Includes LiDAR, mesh networking, and hardware security features
 */
export type DeviceCapabilities = {
    lidarSupported: boolean;
    lidarResolution: number; // in centimeters (0.01cm resolution)
    meshNetworkSupported: boolean;
    meshNetworkLatency: number; // in milliseconds
    vulkanVersion: string; // e.g., "1.3.0"
    tpmVersion: string; // e.g., "2.0"
    maxScanRange: number; // in meters
    maxFleetSize: number; // max 32 devices
    hardwareSecurityLevel: 'HIGH' | 'MEDIUM' | 'LOW';
    scanningFrequency: number; // in Hz
};

/**
 * Enhanced user authentication state with hardware security integration
 */
export type UserAuthState = {
    isAuthenticated: boolean;
    token: string; // JWT token
    tokenExpiry: Date;
    hardwareToken: string; // Hardware-backed security token
    tpmSignature: string; // TPM-based signature
    refreshToken: string;
    lastAuthMethod: 'HARDWARE' | 'OAUTH' | 'MFA';
};

/**
 * Fleet management status type
 */
export type FleetStatus = {
    fleetId: UUID | null;
    role: 'LEADER' | 'MEMBER';
    connectedDevices: number;
    meshLatency: number;
    scanSyncStatus: 'SYNCED' | 'SYNCING' | 'OUT_OF_SYNC';
    lastSyncTime: Date;
};

/**
 * LiDAR scan history type
 */
export type ScanHistory = {
    scanId: UUID;
    timestamp: Date;
    resolution: number;
    pointCount: number;
    coverage: number;
    sharedWithFleet: boolean;
};

/**
 * Enhanced user preferences type
 */
export type UserPreferences = {
    theme: 'LIGHT' | 'DARK' | 'SYSTEM';
    notifications: {
        fleetInvites: boolean;
        scanComplete: boolean;
        securityAlerts: boolean;
    };
    privacy: {
        shareLocation: boolean;
        shareScanData: boolean;
        dataRetentionDays: number;
    };
    performance: {
        lidarQuality: 'HIGH' | 'MEDIUM' | 'LOW';
        meshOptimization: 'LATENCY' | 'BANDWIDTH';
    };
};

/**
 * Comprehensive user profile interface
 */
export interface UserProfile {
    id: UUID;
    displayName: string;
    avatar: string;
    status: keyof UserStatus;
    preferences: UserPreferences;
    deviceCapabilities: DeviceCapabilities;
    fleetStatus: FleetStatus;
    scanHistory: ScanHistory[];
    securityLevel: 'HIGH' | 'STANDARD';
    lastActive: Date;
}

/**
 * Enhanced user authentication data interface
 */
export interface UserAuthData {
    userId: UUID;
    accessToken: string;
    refreshToken: string;
    hardwareToken: string;
    tpmSignature: string;
    expiresAt: Date;
    securityContext: {
        tpmEnabled: boolean;
        mfaEnabled: boolean;
        lastAuthMethod: string;
        securityLevel: string;
    };
}

/**
 * User permission type for RBAC
 */
export type UserPermissions = {
    canManageFleet: boolean;
    canShareScans: boolean;
    canModifySettings: boolean;
    canInviteMembers: boolean;
    maxFleetSize: number;
    scanRetentionDays: number;
};

/**
 * Hardware security state type
 */
export type HardwareSecurityState = {
    tpmAvailable: boolean;
    tpmVersion: string;
    secureBootEnabled: boolean;
    hardwareTokenPresent: boolean;
    integrityLevel: 'HIGH' | 'MEDIUM' | 'LOW';
};