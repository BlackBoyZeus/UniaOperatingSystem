import { injectable } from 'inversify'; // version: 6.0.1
import { Router, Request, Response, NextFunction } from 'express'; // version: 4.18.2
import { rateLimit } from 'express-rate-limit'; // version: 6.7.0
import { Logger } from 'winston'; // version: 3.10.0

import { GameService } from '../../services/game/GameService';
import { 
    IGameState, 
    IEnvironmentState, 
    IStateMetrics 
} from '../../interfaces/game.interface';
import { 
    gameStateSchema, 
    environmentStateSchema, 
    ValidationError 
} from '../validators/game.validator';

// Global constants for rate limiting and performance thresholds
const MAX_ENVIRONMENT_UPDATE_RATE = 30; // 30Hz max update rate
const MAX_FLEET_SIZE = 32;
const MAX_REQUEST_RATE = 100; // requests per minute
const PERFORMANCE_THRESHOLD_MS = 50;

@injectable()
export class GameController {
    private readonly _rateLimiter: any;
    private readonly _performanceMetrics: Map<string, number>;

    constructor(
        private readonly _gameService: GameService,
        private readonly _logger: Logger
    ) {
        this._performanceMetrics = new Map();
        this._rateLimiter = rateLimit({
            windowMs: 60 * 1000, // 1 minute window
            max: MAX_REQUEST_RATE,
            message: 'Too many requests from this device'
        });
    }

    /**
     * Creates a new game session with fleet validation
     */
    public async createSession(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const startTime = performance.now();

        try {
            // Validate request body against schema
            const validationResult = gameStateSchema.safeParse(req.body);
            if (!validationResult.success) {
                return res.status(400).json({
                    error: 'Invalid game state configuration',
                    details: validationResult.error.errors
                });
            }

            // Extract and validate fleet size
            const { fleetId, config } = req.body;
            if (config.fleetSize > MAX_FLEET_SIZE) {
                return res.status(400).json({
                    error: `Fleet size cannot exceed ${MAX_FLEET_SIZE} devices`
                });
            }

            // Create game session
            const gameState = await this._gameService.createSession(
                crypto.randomUUID(),
                fleetId,
                config
            );

            // Record performance metrics
            this._performanceMetrics.set('createSession', performance.now() - startTime);

            this._logger.info('Game session created', {
                sessionId: gameState.sessionId,
                fleetId,
                latency: performance.now() - startTime
            });

            return res.status(201).json(gameState);

        } catch (error) {
            this._logger.error('Failed to create game session', {
                error: error.message,
                latency: performance.now() - startTime
            });
            return next(error);
        }
    }

    /**
     * Joins an existing game session with latency monitoring
     */
    public async joinSession(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const startTime = performance.now();

        try {
            const { sessionId, deviceId } = req.params;
            const { capabilities } = req.body;

            // Validate session parameters
            if (!sessionId || !deviceId) {
                return res.status(400).json({
                    error: 'Missing required parameters'
                });
            }

            // Join session
            await this._gameService.joinGameSession(sessionId, deviceId, capabilities);

            const latency = performance.now() - startTime;
            this._performanceMetrics.set('joinSession', latency);

            if (latency > PERFORMANCE_THRESHOLD_MS) {
                this._logger.warn('High latency in session join', {
                    sessionId,
                    deviceId,
                    latency
                });
            }

            return res.status(200).json({
                success: true,
                latency
            });

        } catch (error) {
            this._logger.error('Failed to join game session', {
                error: error.message,
                latency: performance.now() - startTime
            });
            return next(error);
        }
    }

    /**
     * Updates environment state with rate limiting
     */
    public async updateEnvironment(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const startTime = performance.now();

        try {
            const { sessionId } = req.params;
            const environmentState: IEnvironmentState = req.body;

            // Validate environment state
            const validationResult = environmentStateSchema.safeParse(environmentState);
            if (!validationResult.success) {
                return res.status(400).json({
                    error: 'Invalid environment state',
                    details: validationResult.error.errors
                });
            }

            // Apply rate limiting
            if (this.isRateLimited(sessionId, 'environmentUpdate')) {
                return res.status(429).json({
                    error: `Update rate cannot exceed ${MAX_ENVIRONMENT_UPDATE_RATE}Hz`
                });
            }

            // Process update
            await this._gameService.updateEnvironment(sessionId, environmentState);

            const latency = performance.now() - startTime;
            this._performanceMetrics.set('updateEnvironment', latency);

            return res.status(200).json({
                success: true,
                latency
            });

        } catch (error) {
            this._logger.error('Failed to update environment', {
                error: error.message,
                latency: performance.now() - startTime
            });
            return next(error);
        }
    }

    /**
     * Ends game session with cleanup
     */
    public async endSession(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const startTime = performance.now();

        try {
            const { sessionId } = req.params;

            await this._gameService.endSession(sessionId);

            this._logger.info('Game session ended', {
                sessionId,
                latency: performance.now() - startTime
            });

            return res.status(200).json({
                success: true,
                sessionId
            });

        } catch (error) {
            this._logger.error('Failed to end game session', {
                error: error.message,
                latency: performance.now() - startTime
            });
            return next(error);
        }
    }

    /**
     * Retrieves session performance metrics
     */
    public async getSessionMetrics(req: Request, res: Response, next: NextFunction): Promise<Response> {
        const startTime = performance.now();

        try {
            const { sessionId } = req.params;
            const metrics = await this._gameService.getSessionMetrics(sessionId);

            return res.status(200).json({
                ...metrics,
                controllerLatency: this._performanceMetrics.get('updateEnvironment') || 0,
                requestLatency: performance.now() - startTime
            });

        } catch (error) {
            this._logger.error('Failed to retrieve session metrics', {
                error: error.message,
                latency: performance.now() - startTime
            });
            return next(error);
        }
    }

    private isRateLimited(sessionId: string, operationType: string): boolean {
        const key = `${sessionId}:${operationType}`;
        const now = Date.now();
        const lastUpdate = this._performanceMetrics.get(key) || 0;
        
        if (now - lastUpdate < (1000 / MAX_ENVIRONMENT_UPDATE_RATE)) {
            return true;
        }

        this._performanceMetrics.set(key, now);
        return false;
    }
}