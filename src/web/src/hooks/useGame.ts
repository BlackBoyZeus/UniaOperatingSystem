// External imports with versions for security tracking
import { useState, useEffect, useCallback, useRef, useMemo } from 'react'; // ^18.2.0

// Internal imports
import { useGameContext } from '../contexts/GameContext';
import { GameService } from '../services/game.service';
import {
    IWebGameState,
    GameStates,
    IWebEnvironmentState,
    IWebRenderState,
    IPerformanceMetrics,
    RenderQuality
} from '../interfaces/game.interface';

// Constants for performance optimization
const GAME_UPDATE_INTERVAL = 16; // ~60 FPS target
const PERFORMANCE_MONITOR_INTERVAL = 1000;
const FPS_THRESHOLD = 55;
const MEMORY_THRESHOLD = 0.8;
const POINT_CLOUD_QUALITY_LEVELS = {
    HIGH: 1.0,
    MEDIUM: 0.7,
    LOW: 0.4
} as const;

/**
 * Enhanced interface for game configuration
 */
interface IGameConfig {
    initialQuality?: RenderQuality;
    lidarEnabled?: boolean;
    adaptiveQuality?: boolean;
    maxFleetSize?: number;
}

/**
 * Enhanced custom hook for managing game state, session lifecycle, and performance optimization
 * @param sessionId - Unique identifier for the game session
 * @param config - Game configuration options
 */
export function useGame(sessionId: string, config: IGameConfig = {}) {
    // Context and service initialization
    const { state, dispatch } = useGameContext();
    const gameServiceRef = useRef<GameService | null>(null);
    const frameRequestRef = useRef<number>();
    const performanceMonitorRef = useRef<NodeJS.Timeout>();

    // Local state for performance metrics
    const [performanceMetrics, setPerformanceMetrics] = useState<IPerformanceMetrics>({
        fps: 60,
        frameTime: 0,
        memoryUsage: 0
    });

    // Initialize game service with configuration
    useEffect(() => {
        gameServiceRef.current = new GameService();
        return () => {
            gameServiceRef.current?.dispose();
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }
            if (performanceMonitorRef.current) {
                clearInterval(performanceMonitorRef.current);
            }
        };
    }, []);

    // Performance monitoring setup
    useEffect(() => {
        let lastFrameTime = performance.now();

        const monitorPerformance = () => {
            const currentTime = performance.now();
            const frameTime = currentTime - lastFrameTime;
            const fps = Math.round(1000 / frameTime);
            const memory = (performance as any).memory?.usedJSHeapSize / 
                          (performance as any).memory?.jsHeapSizeLimit || 0;

            setPerformanceMetrics({
                fps,
                frameTime,
                memoryUsage: memory
            });

            // Adaptive quality management
            if (config.adaptiveQuality && state.state === GameStates.RUNNING) {
                if (fps < FPS_THRESHOLD || memory > MEMORY_THRESHOLD) {
                    optimizePerformance(fps, memory);
                }
            }

            lastFrameTime = currentTime;
            frameRequestRef.current = requestAnimationFrame(monitorPerformance);
        };

        frameRequestRef.current = requestAnimationFrame(monitorPerformance);
        return () => {
            if (frameRequestRef.current) {
                cancelAnimationFrame(frameRequestRef.current);
            }
        };
    }, [config.adaptiveQuality, state.state]);

    // Optimize performance based on metrics
    const optimizePerformance = useCallback((fps: number, memory: number) => {
        if (!gameServiceRef.current) return;

        let qualityLevel = POINT_CLOUD_QUALITY_LEVELS.HIGH;
        if (fps < FPS_THRESHOLD * 0.8 || memory > MEMORY_THRESHOLD * 0.9) {
            qualityLevel = POINT_CLOUD_QUALITY_LEVELS.MEDIUM;
        } else if (fps < FPS_THRESHOLD * 0.6 || memory > MEMORY_THRESHOLD) {
            qualityLevel = POINT_CLOUD_QUALITY_LEVELS.LOW;
        }

        gameServiceRef.current.optimizePointCloud(qualityLevel);
    }, []);

    // Start game session
    const startGame = useCallback(async () => {
        if (!gameServiceRef.current) return;

        try {
            await gameServiceRef.current.startGameSession(sessionId, {
                quality: config.initialQuality || RenderQuality.HIGH,
                lidarEnabled: config.lidarEnabled ?? true,
                maxFleetSize: config.maxFleetSize || 32
            });

            dispatch({ type: 'START_SESSION', payload: { sessionId } });
        } catch (error) {
            console.error('Failed to start game session:', error);
            dispatch({ type: 'SET_ERROR', payload: error.message });
        }
    }, [sessionId, config, dispatch]);

    // Pause game
    const pauseGame = useCallback(() => {
        if (state.state !== GameStates.RUNNING) return;
        dispatch({ type: 'UPDATE_GAME_STATE', payload: GameStates.PAUSED });
    }, [state.state, dispatch]);

    // Resume game
    const resumeGame = useCallback(() => {
        if (state.state !== GameStates.PAUSED) return;
        dispatch({ type: 'UPDATE_GAME_STATE', payload: GameStates.RUNNING });
    }, [state.state, dispatch]);

    // End game session
    const endGame = useCallback(async () => {
        if (!gameServiceRef.current) return;

        try {
            await gameServiceRef.current.endGameSession(sessionId);
            dispatch({ type: 'END_SESSION' });
        } catch (error) {
            console.error('Failed to end game session:', error);
            dispatch({ type: 'SET_ERROR', payload: error.message });
        }
    }, [sessionId, dispatch]);

    // Update environment with optimized processing
    const updateEnvironment = useCallback((data: IWebEnvironmentState) => {
        if (!gameServiceRef.current || state.state !== GameStates.RUNNING) return;

        try {
            const optimizedData = gameServiceRef.current.updateEnvironment(
                data,
                performanceMetrics.fps
            );
            dispatch({ type: 'UPDATE_ENVIRONMENT', payload: optimizedData });
        } catch (error) {
            console.error('Failed to update environment:', error);
            dispatch({ type: 'SET_ERROR', payload: error.message });
        }
    }, [state.state, performanceMetrics.fps, dispatch]);

    // Memoized game state for consumers
    const gameState = useMemo(() => ({
        ...state,
        performanceMetrics
    }), [state, performanceMetrics]);

    return {
        gameState,
        startGame,
        pauseGame,
        resumeGame,
        endGame,
        updateEnvironment,
        performanceMetrics
    };
}

export default useGame;