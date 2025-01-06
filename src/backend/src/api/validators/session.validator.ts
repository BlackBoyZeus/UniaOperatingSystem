/**
 * @file Advanced session validator for TALD UNIA platform
 * @version 1.0.0
 */

import { z } from 'zod'; // v3.22.2
import { ISession, ISessionConfig, ISessionState, SessionStatus } from '../../interfaces/session.interface';
import { validateFleetConfiguration } from '../../utils/validation.utils';

// Global constants for session validation
const MAX_SESSION_PARTICIPANTS = 32;
const DEFAULT_SCAN_RATE = 30;
const MAX_SESSION_DURATION = 14400000; // 4 hours in milliseconds
const MAX_LATENCY = 50;
const MIN_SCAN_QUALITY = 0.95;
const PERFORMANCE_CHECK_INTERVAL = 1000;

/**
 * Enhanced error type for session validation failures
 */
class SessionValidationError extends Error {
    constructor(
        public code: string,
        message: string,
        public details: any,
        public metrics: object,
        public timestamp: string
    ) {
        super(message);
        this.name = 'SessionValidationError';
    }
}

/**
 * Enhanced Zod schema for session configuration validation
 */
export const sessionConfigSchema = z.object({
    maxParticipants: z.number().min(1).max(MAX_SESSION_PARTICIPANTS),
    networkConfig: z.object({
        meshTopology: z.enum(['full', 'star', 'ring']),
        maxLatency: z.number().max(MAX_LATENCY),
        syncInterval: z.number().min(20).max(100),
        compressionEnabled: z.boolean(),
        encryptionEnabled: z.boolean()
    }),
    scanRate: z.number().min(1).max(DEFAULT_SCAN_RATE),
    performanceThresholds: z.object({
        minScanQuality: z.number().min(MIN_SCAN_QUALITY),
        maxLatency: z.number().max(MAX_LATENCY),
        minFrameRate: z.number().min(60),
        maxMemoryUsage: z.number(),
        maxCpuUsage: z.number()
    }),
    autoRecoveryEnabled: z.boolean(),
    stateValidation: z.boolean()
});

/**
 * Advanced Zod schema for session state validation
 */
export const sessionStateSchema = z.object({
    status: z.nativeEnum(SessionStatus),
    activeParticipants: z.number().min(1).max(MAX_SESSION_PARTICIPANTS),
    averageLatency: z.number().max(MAX_LATENCY),
    lastUpdate: z.date(),
    performanceMetrics: z.object({
        averageLatency: z.number().max(MAX_LATENCY),
        packetLoss: z.number().min(0).max(1),
        syncRate: z.number(),
        cpuUsage: z.number().min(0).max(100),
        memoryUsage: z.number().min(0),
        batteryLevel: z.number().min(0).max(100),
        networkBandwidth: z.number().min(0),
        scanQuality: z.number().min(MIN_SCAN_QUALITY),
        frameRate: z.number().min(60),
        lastUpdate: z.number()
    }),
    errorCount: z.number().min(0),
    warningCount: z.number().min(0),
    recoveryAttempts: z.number().min(0)
});

/**
 * Enhanced session configuration validation with security checks
 */
export async function validateSessionConfig(config: ISessionConfig): Promise<boolean> {
    try {
        // Validate basic configuration structure
        await sessionConfigSchema.parseAsync(config);

        // Validate fleet configuration
        const fleetValidation = await validateFleetConfiguration({
            maxDevices: config.maxParticipants,
            topology: config.networkConfig.meshTopology,
            networkStats: {
                averageLatency: config.networkConfig.maxLatency
            }
        }, 'member', config.networkConfig);

        if (!fleetValidation.valid) {
            throw new SessionValidationError(
                'FLEET_CONFIG_ERROR',
                'Fleet configuration validation failed',
                fleetValidation.errors,
                { timestamp: Date.now() },
                new Date().toISOString()
            );
        }

        // Validate performance thresholds
        if (config.performanceThresholds.maxLatency > MAX_LATENCY) {
            throw new SessionValidationError(
                'LATENCY_THRESHOLD_ERROR',
                `Latency threshold exceeds maximum allowed value of ${MAX_LATENCY}ms`,
                { configured: config.performanceThresholds.maxLatency, maximum: MAX_LATENCY },
                { timestamp: Date.now() },
                new Date().toISOString()
            );
        }

        // Validate scan rate configuration
        if (config.scanRate > DEFAULT_SCAN_RATE) {
            throw new SessionValidationError(
                'SCAN_RATE_ERROR',
                `Scan rate exceeds maximum allowed value of ${DEFAULT_SCAN_RATE}Hz`,
                { configured: config.scanRate, maximum: DEFAULT_SCAN_RATE },
                { timestamp: Date.now() },
                new Date().toISOString()
            );
        }

        return true;
    } catch (error) {
        if (error instanceof SessionValidationError) {
            throw error;
        }
        throw new SessionValidationError(
            'CONFIG_VALIDATION_ERROR',
            'Session configuration validation failed',
            error,
            { timestamp: Date.now() },
            new Date().toISOString()
        );
    }
}

/**
 * Comprehensive session state validation with performance monitoring
 */
export async function validateSessionState(state: ISessionState): Promise<boolean> {
    try {
        // Validate basic state structure
        await sessionStateSchema.parseAsync(state);

        // Validate participant count
        if (state.activeParticipants > MAX_SESSION_PARTICIPANTS) {
            throw new SessionValidationError(
                'PARTICIPANT_LIMIT_ERROR',
                `Active participants exceed maximum limit of ${MAX_SESSION_PARTICIPANTS}`,
                { current: state.activeParticipants, maximum: MAX_SESSION_PARTICIPANTS },
                { timestamp: Date.now() },
                new Date().toISOString()
            );
        }

        // Validate network performance
        if (state.performanceMetrics.averageLatency > MAX_LATENCY) {
            throw new SessionValidationError(
                'LATENCY_ERROR',
                `Network latency exceeds maximum threshold of ${MAX_LATENCY}ms`,
                { current: state.performanceMetrics.averageLatency, maximum: MAX_LATENCY },
                state.performanceMetrics,
                new Date().toISOString()
            );
        }

        // Validate scan quality
        if (state.performanceMetrics.scanQuality < MIN_SCAN_QUALITY) {
            throw new SessionValidationError(
                'SCAN_QUALITY_ERROR',
                `Scan quality below minimum threshold of ${MIN_SCAN_QUALITY}`,
                { current: state.performanceMetrics.scanQuality, minimum: MIN_SCAN_QUALITY },
                state.performanceMetrics,
                new Date().toISOString()
            );
        }

        // Validate state freshness
        const stateAge = Date.now() - state.lastUpdate.getTime();
        if (stateAge > PERFORMANCE_CHECK_INTERVAL) {
            throw new SessionValidationError(
                'STATE_FRESHNESS_ERROR',
                'Session state update interval exceeded',
                { stateAge, maximum: PERFORMANCE_CHECK_INTERVAL },
                state.performanceMetrics,
                new Date().toISOString()
            );
        }

        // Validate recovery state
        if (state.recoveryAttempts > 3 && state.status !== SessionStatus.TERMINATED) {
            throw new SessionValidationError(
                'RECOVERY_LIMIT_ERROR',
                'Maximum recovery attempts exceeded',
                { attempts: state.recoveryAttempts, maximum: 3 },
                state.performanceMetrics,
                new Date().toISOString()
            );
        }

        return true;
    } catch (error) {
        if (error instanceof SessionValidationError) {
            throw error;
        }
        throw new SessionValidationError(
            'STATE_VALIDATION_ERROR',
            'Session state validation failed',
            error,
            { timestamp: Date.now() },
            new Date().toISOString()
        );
    }
}