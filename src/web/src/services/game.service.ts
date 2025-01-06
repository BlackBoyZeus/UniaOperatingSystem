// External imports with versions for security tracking
import { BehaviorSubject, ReplaySubject } from 'rxjs'; // ^7.8.0
import { throttle, debounce } from 'lodash'; // ^4.17.21
import * as Automerge from 'automerge'; // ^1.0.1

// Internal imports
import { ApiService } from './api.service';
import { 
    IWebGameState, 
    IWebEnvironmentState, 
    IWebRenderState, 
    GameStates, 
    GameEvents, 
    RenderQuality 
} from '../interfaces/game.interface';
import { 
    calculateFPS, 
    validateGameState, 
    processEnvironmentData, 
    optimizeRenderConfig, 
    handleCRDTMerge 
} from '../utils/game.utils';

// Global constants from specification
const STATE_UPDATE_INTERVAL = 16; // 60 FPS target
const MAX_RETRY_ATTEMPTS = 3;
const ENVIRONMENT_SYNC_INTERVAL = 100;
const MAX_FLEET_SIZE = 32;
const RENDER_QUALITY_THRESHOLD = 58; // FPS threshold for quality adjustment
const MEMORY_THRESHOLD_MB = 3800;

/**
 * Enhanced GameService for managing game state and real-time synchronization
 * Implements CRDT-based state management and performance optimization
 */
export class GameService {
    private gameState$: BehaviorSubject<IWebGameState>;
    private crdtState$: ReplaySubject<Automerge.Doc<any>>;
    private environmentSync$: BehaviorSubject<IWebEnvironmentState | null>;
    private performanceMetrics$: BehaviorSubject<{ fps: number; memory: number }>;
    private lastFrameTime: number = 0;
    private crdtDoc: Automerge.Doc<any>;
    private syncInterval: number | null = null;
    private retryCount: number = 0;

    constructor(
        private readonly apiService: ApiService
    ) {
        this.initializeService();
    }

    /**
     * Initializes service state and observables
     */
    private initializeService(): void {
        this.gameState$ = new BehaviorSubject<IWebGameState>({
            gameId: '',
            sessionId: '',
            state: GameStates.INITIALIZING,
            environmentData: null,
            renderState: {
                resolution: { width: window.innerWidth, height: window.innerHeight },
                quality: RenderQuality.HIGH,
                lidarOverlayEnabled: true
            },
            fps: 60
        });

        this.crdtState$ = new ReplaySubject(1);
        this.environmentSync$ = new BehaviorSubject<IWebEnvironmentState | null>(null);
        this.performanceMetrics$ = new BehaviorSubject({ fps: 60, memory: 0 });
        this.crdtDoc = Automerge.init();
    }

    /**
     * Starts a new game session with enhanced state management
     */
    public async startGame(gameId: string, sessionId: string): Promise<void> {
        try {
            // Initialize game state with CRDT support
            this.crdtDoc = Automerge.change(this.crdtDoc, 'Initialize game', doc => {
                doc.gameId = gameId;
                doc.sessionId = sessionId;
                doc.state = GameStates.INITIALIZING;
            });

            // Set up WebRTC connection for P2P communication
            await this.apiService.connectWebRTC();

            // Start game loop and monitoring
            this.startGameLoop();
            this.startEnvironmentSync();
            this.monitorPerformance();

            // Update game state
            this.updateGameState({
                ...this.gameState$.value,
                gameId,
                sessionId,
                state: GameStates.RUNNING
            });

        } catch (error) {
            console.error('Failed to start game:', error);
            this.handleError(error);
        }
    }

    /**
     * Updates game state with CRDT synchronization
     */
    private updateGameState(newState: IWebGameState): void {
        try {
            // Update CRDT document
            this.crdtDoc = Automerge.change(this.crdtDoc, 'Update state', doc => {
                Object.assign(doc, newState);
            });

            // Validate state update
            const { isValid, conflicts } = validateGameState(newState, this.crdtDoc);
            if (!isValid) {
                throw new Error(`Invalid state update: ${conflicts.join(', ')}`);
            }

            // Emit updates
            this.gameState$.next(newState);
            this.crdtState$.next(this.crdtDoc);

        } catch (error) {
            console.error('State update failed:', error);
            this.handleError(error);
        }
    }

    /**
     * Handles environment updates with performance optimization
     */
    private handleEnvironmentUpdate = throttle(
        async (environmentData: IWebEnvironmentState): Promise<void> => {
            try {
                const currentFPS = this.performanceMetrics$.value.fps;
                
                // Process and optimize environment data
                const optimizedData = processEnvironmentData(environmentData, currentFPS);
                
                // Update game state with new environment data
                this.updateGameState({
                    ...this.gameState$.value,
                    environmentData: optimizedData
                });

                // Sync with fleet if in multiplayer
                if (this.gameState$.value.state === GameStates.RUNNING) {
                    await this.syncWithFleet(optimizedData);
                }

            } catch (error) {
                console.error('Environment update failed:', error);
                this.handleError(error);
            }
        },
        STATE_UPDATE_INTERVAL
    );

    /**
     * Synchronizes state with fleet using CRDT
     */
    private async syncWithFleet(environmentData: IWebEnvironmentState): Promise<void> {
        try {
            const changes = Automerge.getChanges(this.crdtDoc, Automerge.init());
            await this.apiService.emit('fleet:sync', {
                changes,
                environmentData,
                timestamp: Date.now()
            });
        } catch (error) {
            console.error('Fleet sync failed:', error);
            this.retrySync();
        }
    }

    /**
     * Starts the main game loop with performance monitoring
     */
    private startGameLoop(): void {
        const gameLoop = () => {
            const currentTime = performance.now();
            
            // Calculate FPS and check performance
            const { fps, shouldOptimize } = calculateFPS(
                currentTime,
                this.lastFrameTime,
                this.gameState$.value.renderState
            );

            // Optimize render quality if needed
            if (shouldOptimize) {
                this.optimizeRenderQuality(fps);
            }

            this.lastFrameTime = currentTime;
            this.performanceMetrics$.next({ ...this.performanceMetrics$.value, fps });

            // Continue loop if game is running
            if (this.gameState$.value.state === GameStates.RUNNING) {
                requestAnimationFrame(gameLoop);
            }
        };

        requestAnimationFrame(gameLoop);
    }

    /**
     * Starts environment synchronization loop
     */
    private startEnvironmentSync(): void {
        this.syncInterval = window.setInterval(() => {
            if (this.environmentSync$.value) {
                this.handleEnvironmentUpdate(this.environmentSync$.value);
            }
        }, ENVIRONMENT_SYNC_INTERVAL);
    }

    /**
     * Monitors and optimizes performance
     */
    private monitorPerformance(): void {
        setInterval(() => {
            const memory = (performance as any).memory?.usedJSHeapSize / (1024 * 1024) || 0;
            
            // Check memory usage
            if (memory > MEMORY_THRESHOLD_MB) {
                this.optimizeMemoryUsage();
            }

            this.performanceMetrics$.next({
                ...this.performanceMetrics$.value,
                memory
            });
        }, 1000);
    }

    /**
     * Optimizes render quality based on performance
     */
    private optimizeRenderQuality(currentFPS: number): void {
        const currentState = this.gameState$.value;
        let newQuality = currentState.renderState.quality;

        if (currentFPS < RENDER_QUALITY_THRESHOLD) {
            newQuality = currentState.renderState.quality === RenderQuality.HIGH 
                ? RenderQuality.MEDIUM 
                : RenderQuality.LOW;
        }

        this.updateGameState({
            ...currentState,
            renderState: {
                ...currentState.renderState,
                quality: newQuality
            }
        });
    }

    /**
     * Optimizes memory usage when approaching threshold
     */
    private optimizeMemoryUsage(): void {
        this.environmentSync$.value = null;
        this.crdtDoc = Automerge.clone(this.crdtDoc);
        global.gc?.();
    }

    /**
     * Handles retry logic for failed operations
     */
    private retrySync(): void {
        if (this.retryCount < MAX_RETRY_ATTEMPTS) {
            this.retryCount++;
            setTimeout(() => {
                this.syncWithFleet(this.environmentSync$.value!);
            }, Math.pow(2, this.retryCount) * 1000);
        } else {
            this.handleError(new Error('Max retry attempts exceeded'));
        }
    }

    /**
     * Handles errors and updates game state accordingly
     */
    private handleError(error: Error): void {
        console.error('GameService error:', error);
        this.updateGameState({
            ...this.gameState$.value,
            state: GameStates.ERROR
        });
    }

    /**
     * Cleans up resources on service destruction
     */
    public dispose(): void {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
        }
        this.gameState$.complete();
        this.crdtState$.complete();
        this.environmentSync$.complete();
        this.performanceMetrics$.complete();
    }
}

export default GameService;