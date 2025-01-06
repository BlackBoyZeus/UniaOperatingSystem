import { injectable, inject } from 'inversify'; // version: 6.0.1
import { EventEmitter } from 'events'; // version: 3.3.0
import { PerformanceMonitor } from '@monitoring/performance'; // version: 1.0.0

import { GameEngine } from '../../core/game/GameEngine';
import { GameState } from '../../core/game/GameState';
import {
    IGameState,
    IEnvironmentState,
    IPhysicsState,
    Vector3,
    IPerformanceMetrics
} from '../../interfaces/game.interface';

// Global constants for game service configuration
const STATE_SYNC_INTERVAL = 50; // 50ms sync interval
const MAX_RETRY_ATTEMPTS = 3;
const SESSION_TIMEOUT = 300000; // 5 minutes
const THERMAL_THRESHOLD_CELSIUS = 75;
const BATTERY_THRESHOLD_PERCENT = 20;

// Power mode enumeration
enum PowerMode {
    PERFORMANCE = 'PERFORMANCE',
    BALANCED = 'BALANCED',
    POWER_SAVE = 'POWER_SAVE'
}

// Thermal state enumeration
enum ThermalState {
    NORMAL = 'NORMAL',
    WARNING = 'WARNING',
    CRITICAL = 'CRITICAL'
}

// Session configuration interface
interface SessionConfig {
    fleetSize: number;
    powerMode: PowerMode;
    thermalPolicy: {
        warningThreshold: number;
        criticalThreshold: number;
    };
}

// Game session interface
interface GameSession {
    id: string;
    fleetId: string;
    state: GameState;
    config: SessionConfig;
    startTime: number;
    lastUpdate: number;
    metrics: IPerformanceMetrics;
}

@injectable()
export class GameService extends EventEmitter {
    private readonly _gameEngine: GameEngine;
    private readonly _gameState: GameState;
    private readonly _performanceMonitor: PerformanceMonitor;
    private readonly _activeSessions: Map<string, GameSession>;
    private _isRunning: boolean;
    private _powerMode: PowerMode;
    private _thermalState: ThermalState;
    private _syncInterval: NodeJS.Timeout | null;

    constructor(
        @inject(GameEngine) gameEngine: GameEngine,
        @inject(GameState) gameState: GameState,
        @inject(PerformanceMonitor) performanceMonitor: PerformanceMonitor
    ) {
        super();
        this._gameEngine = gameEngine;
        this._gameState = gameState;
        this._performanceMonitor = performanceMonitor;
        this._activeSessions = new Map();
        this._isRunning = false;
        this._powerMode = PowerMode.BALANCED;
        this._thermalState = ThermalState.NORMAL;
        this._syncInterval = null;

        this.setupEventHandlers();
    }

    /**
     * Creates a new game session with enhanced validation and monitoring
     */
    public async createSession(
        sessionId: string,
        fleetId: string,
        config: SessionConfig
    ): Promise<IGameState> {
        try {
            // Validate session parameters
            this.validateSessionParameters(sessionId, fleetId, config);

            // Check system resources
            await this.checkSystemResources();

            // Initialize game engine with power profile
            await this._gameEngine.start(this.determinePowerMode(config.powerMode));

            // Create new game state instance
            const gameState = new GameState(sessionId, fleetId, config.fleetSize);

            // Create session record
            const session: GameSession = {
                id: sessionId,
                fleetId,
                state: gameState,
                config,
                startTime: Date.now(),
                lastUpdate: Date.now(),
                metrics: this.initializeMetrics()
            };

            // Store session
            this._activeSessions.set(sessionId, session);

            // Start monitoring
            this.startSessionMonitoring(session);

            return gameState.getState();
        } catch (error) {
            this.handleError('createSession', error);
            throw error;
        }
    }

    /**
     * Ends a game session and cleans up resources
     */
    public async endSession(sessionId: string): Promise<void> {
        try {
            const session = this.getSession(sessionId);
            
            // Stop monitoring
            this.stopSessionMonitoring(session);

            // Cleanup resources
            session.state.dispose();
            this._activeSessions.delete(sessionId);

            // Emit session end event
            this.emit('sessionEnded', {
                sessionId,
                duration: Date.now() - session.startTime,
                metrics: session.metrics
            });
        } catch (error) {
            this.handleError('endSession', error);
            throw error;
        }
    }

    /**
     * Processes environment updates with power and thermal management
     */
    public async processEnvironmentUpdate(
        sessionId: string,
        environmentState: IEnvironmentState
    ): Promise<void> {
        try {
            const session = this.getSession(sessionId);
            const startTime = performance.now();

            // Check thermal state
            this.checkThermalState();

            // Process update with power consideration
            await session.state.updateEnvironment(environmentState);

            // Update metrics
            session.metrics.lidarProcessingLatency = performance.now() - startTime;
            session.lastUpdate = Date.now();

            this.emit('environmentUpdated', {
                sessionId,
                metrics: session.metrics,
                thermalState: this._thermalState
            });
        } catch (error) {
            this.handleError('processEnvironmentUpdate', error);
            throw error;
        }
    }

    /**
     * Synchronizes game state across fleet with retry mechanism
     */
    public async synchronizeState(
        sessionId: string,
        stateUpdate: Partial<IGameState>
    ): Promise<void> {
        try {
            const session = this.getSession(sessionId);
            const startTime = performance.now();

            // Apply state update with retry mechanism
            let attempts = 0;
            while (attempts < MAX_RETRY_ATTEMPTS) {
                try {
                    await session.state.applyChange({
                        documentId: session.id,
                        operation: 'UPDATE',
                        timestamp: Date.now(),
                        retryCount: attempts
                    });
                    break;
                } catch (error) {
                    attempts++;
                    if (attempts === MAX_RETRY_ATTEMPTS) throw error;
                    await this.delay(Math.pow(2, attempts) * 50); // Exponential backoff
                }
            }

            // Update metrics
            session.metrics.fleetSyncLatency = performance.now() - startTime;
            session.lastUpdate = Date.now();

            this.emit('stateSynchronized', {
                sessionId,
                metrics: session.metrics
            });
        } catch (error) {
            this.handleError('synchronizeState', error);
            throw error;
        }
    }

    /**
     * Retrieves current session state with validation
     */
    public getSessionState(sessionId: string): IGameState {
        try {
            const session = this.getSession(sessionId);
            return session.state.getState();
        } catch (error) {
            this.handleError('getSessionState', error);
            throw error;
        }
    }

    /**
     * Updates power mode with thermal consideration
     */
    public setPowerMode(sessionId: string, mode: PowerMode): void {
        try {
            const session = this.getSession(sessionId);
            this._powerMode = mode;
            this._gameEngine.setPowerMode(mode);
            
            this.emit('powerModeChanged', {
                sessionId,
                mode,
                thermalState: this._thermalState
            });
        } catch (error) {
            this.handleError('setPowerMode', error);
            throw error;
        }
    }

    private setupEventHandlers(): void {
        this._gameEngine.on('error', this.handleEngineError.bind(this));
        this._gameEngine.on('thermal-warning', this.handleThermalWarning.bind(this));
        this._gameEngine.on('thermal-critical', this.handleThermalCritical.bind(this));
        this._performanceMonitor.on('metrics', this.handlePerformanceMetrics.bind(this));
    }

    private validateSessionParameters(
        sessionId: string,
        fleetId: string,
        config: SessionConfig
    ): void {
        if (this._activeSessions.has(sessionId)) {
            throw new Error(`Session ${sessionId} already exists`);
        }
        if (config.fleetSize > 32) {
            throw new Error('Fleet size cannot exceed 32 devices');
        }
    }

    private async checkSystemResources(): Promise<void> {
        const metrics = await this._performanceMonitor.getMetrics();
        if (metrics.temperature > THERMAL_THRESHOLD_CELSIUS) {
            throw new Error('System temperature too high');
        }
        if (metrics.batteryLevel < BATTERY_THRESHOLD_PERCENT) {
            throw new Error('Battery level too low');
        }
    }

    private getSession(sessionId: string): GameSession {
        const session = this._activeSessions.get(sessionId);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }
        return session;
    }

    private startSessionMonitoring(session: GameSession): void {
        this._syncInterval = setInterval(() => {
            this.monitorSession(session);
        }, STATE_SYNC_INTERVAL);
    }

    private stopSessionMonitoring(session: GameSession): void {
        if (this._syncInterval) {
            clearInterval(this._syncInterval);
            this._syncInterval = null;
        }
    }

    private async monitorSession(session: GameSession): Promise<void> {
        try {
            // Check session timeout
            if (Date.now() - session.lastUpdate > SESSION_TIMEOUT) {
                await this.endSession(session.id);
                return;
            }

            // Update performance metrics
            const metrics = await this._performanceMonitor.getMetrics();
            session.metrics = {
                ...session.metrics,
                ...metrics
            };

            // Check thermal state
            this.checkThermalState();

        } catch (error) {
            this.handleError('monitorSession', error);
        }
    }

    private checkThermalState(): void {
        const temperature = this._performanceMonitor.getTemperature();
        if (temperature > this.getThermalThreshold()) {
            this.handleThermalWarning(temperature);
        }
    }

    private handleEngineError(error: Error): void {
        this.emit('error', {
            component: 'GameEngine',
            error: error.message,
            timestamp: Date.now()
        });
    }

    private handleThermalWarning(temperature: number): void {
        this._thermalState = ThermalState.WARNING;
        this.setPowerMode(PowerMode.BALANCED);
        this.emit('thermalWarning', { temperature });
    }

    private handleThermalCritical(temperature: number): void {
        this._thermalState = ThermalState.CRITICAL;
        this.setPowerMode(PowerMode.POWER_SAVE);
        this.emit('thermalCritical', { temperature });
    }

    private handlePerformanceMetrics(metrics: IPerformanceMetrics): void {
        this.emit('metrics', metrics);
    }

    private handleError(operation: string, error: Error): void {
        this.emit('error', {
            operation,
            error: error.message,
            timestamp: Date.now()
        });
    }

    private initializeMetrics(): IPerformanceMetrics {
        return {
            stateUpdateLatency: 0,
            lidarProcessingLatency: 0,
            physicsSimulationLatency: 0,
            fleetSyncLatency: 0
        };
    }

    private determinePowerMode(configMode: PowerMode): PowerMode {
        if (this._thermalState === ThermalState.CRITICAL) {
            return PowerMode.POWER_SAVE;
        }
        return configMode;
    }

    private getThermalThreshold(): number {
        return this._powerMode === PowerMode.PERFORMANCE ? 
            THERMAL_THRESHOLD_CELSIUS - 5 : 
            THERMAL_THRESHOLD_CELSIUS;
    }

    private delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}