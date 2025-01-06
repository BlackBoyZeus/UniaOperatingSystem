import { injectable } from 'inversify';
import { Request, Response } from 'express';
import { Logger } from 'winston';
import { rateLimit } from 'express-rate-limit';

import {
    ISession,
    ISessionConfig,
    ISessionState,
    SessionStatus,
    ISessionMetrics,
    IFleetHealth
} from '../../interfaces/session.interface';

import { SessionService } from '../../services/session/SessionService';
import {
    validateSessionConfig,
    validateSessionState,
    validateFleetSize
} from '../validators/session.validator';

@injectable()
export class SessionController {
    private readonly sessionService: SessionService;
    private readonly logger: Logger;
    private readonly metricsCollector: any;

    // Rate limiters for different endpoints
    private readonly createSessionLimiter = rateLimit({
        windowMs: 60000, // 1 minute
        max: 10 // 10 requests per minute
    });

    private readonly updateStateLimiter = rateLimit({
        windowMs: 1000, // 1 second
        max: 20 // 20 updates per second
    });

    constructor(
        sessionService: SessionService,
        logger: Logger,
        metricsCollector: any
    ) {
        this.sessionService = sessionService;
        this.logger = logger;
        this.metricsCollector = metricsCollector;
    }

    /**
     * Creates a new gaming session with enhanced validation and monitoring
     */
    public async createSession(req: Request, res: Response): Promise<Response> {
        const startTime = Date.now();
        try {
            // Validate session configuration
            const config: ISessionConfig = req.body;
            await validateSessionConfig(config);

            // Validate fleet size constraints
            if (config.maxParticipants > 32) {
                return res.status(400).json({
                    error: 'Fleet size cannot exceed 32 devices',
                    code: 'FLEET_SIZE_ERROR'
                });
            }

            // Create session with monitoring
            const session = await this.sessionService.createSession(config);

            // Track session creation metrics
            this.metricsCollector.trackMetric('session.creation.latency', Date.now() - startTime);
            this.metricsCollector.incrementCounter('session.created');

            this.logger.info('Session created successfully', {
                sessionId: session.sessionId,
                config: config,
                duration: Date.now() - startTime
            });

            return res.status(201).json({
                sessionId: session.sessionId,
                config: session.config,
                state: session.state
            });

        } catch (error) {
            this.logger.error('Failed to create session', {
                error: error.message,
                config: req.body,
                duration: Date.now() - startTime
            });

            return res.status(500).json({
                error: 'Failed to create session',
                details: error.message,
                code: 'SESSION_CREATION_ERROR'
            });
        }
    }

    /**
     * Updates session state with CRDT conflict resolution and latency monitoring
     */
    public async updateSessionState(req: Request, res: Response): Promise<Response> {
        const startTime = Date.now();
        const { sessionId } = req.params;
        const newState: ISessionState = req.body;

        try {
            // Validate session state
            await validateSessionState(newState);

            // Update state with latency tracking
            const latency = Date.now() - newState.lastUpdate.getTime();
            if (latency > 50) {
                this.logger.warn('High state update latency detected', {
                    sessionId,
                    latency,
                    threshold: 50
                });
            }

            const updatedState = await this.sessionService.updateSessionState(sessionId, newState);

            // Track performance metrics
            this.metricsCollector.trackMetric('session.state.updateLatency', Date.now() - startTime);
            this.metricsCollector.trackMetric('session.state.networkLatency', latency);

            return res.status(200).json({
                sessionId,
                state: updatedState,
                performance: {
                    updateLatency: Date.now() - startTime,
                    networkLatency: latency
                }
            });

        } catch (error) {
            this.logger.error('Failed to update session state', {
                sessionId,
                error: error.message,
                duration: Date.now() - startTime
            });

            return res.status(500).json({
                error: 'Failed to update session state',
                details: error.message,
                code: 'STATE_UPDATE_ERROR'
            });
        }
    }

    /**
     * Checks session and fleet health with comprehensive monitoring
     */
    public async checkSessionHealth(req: Request, res: Response): Promise<Response> {
        const startTime = Date.now();
        const { sessionId } = req.params;

        try {
            // Get session health metrics
            const health = await this.sessionService.validateSessionHealth(sessionId);

            // Check fleet performance
            const fleetHealth = await this.sessionService.monitorLatency(sessionId);
            
            // Generate health report
            const healthReport = {
                sessionId,
                status: health.status,
                fleetHealth: {
                    averageLatency: fleetHealth.averageLatency,
                    participantCount: fleetHealth.participantCount,
                    networkQuality: fleetHealth.networkQuality
                },
                performance: {
                    cpuUsage: health.performance.cpuUsage,
                    memoryUsage: health.performance.memoryUsage,
                    frameRate: health.performance.frameRate,
                    scanQuality: health.performance.scanQuality
                },
                timestamp: Date.now()
            };

            // Track health check metrics
            this.metricsCollector.trackMetric('session.health.checkLatency', Date.now() - startTime);
            this.metricsCollector.trackMetric('session.health.fleetLatency', fleetHealth.averageLatency);

            return res.status(200).json(healthReport);

        } catch (error) {
            this.logger.error('Failed to check session health', {
                sessionId,
                error: error.message,
                duration: Date.now() - startTime
            });

            return res.status(500).json({
                error: 'Failed to check session health',
                details: error.message,
                code: 'HEALTH_CHECK_ERROR'
            });
        }
    }

    /**
     * Handles session recovery with automatic failover
     */
    public async recoverSession(req: Request, res: Response): Promise<Response> {
        const startTime = Date.now();
        const { sessionId } = req.params;

        try {
            const recoveryResult = await this.sessionService.recoverSession(sessionId);

            this.logger.info('Session recovery completed', {
                sessionId,
                result: recoveryResult,
                duration: Date.now() - startTime
            });

            return res.status(200).json({
                sessionId,
                recoveryStatus: recoveryResult.status,
                restoredState: recoveryResult.state
            });

        } catch (error) {
            this.logger.error('Failed to recover session', {
                sessionId,
                error: error.message,
                duration: Date.now() - startTime
            });

            return res.status(500).json({
                error: 'Failed to recover session',
                details: error.message,
                code: 'RECOVERY_ERROR'
            });
        }
    }
}

export default SessionController;