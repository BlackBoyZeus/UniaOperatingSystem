// External imports with versions for security tracking
import React, { createContext, useContext, useReducer, useEffect, useCallback, useMemo } from 'react'; // ^18.2.0
import * as Automerge from 'automerge'; // ^2.0.0

// Internal imports
import { GameService } from '../services/game.service';
import {
    IWebGameState,
    GameStates,
    GameEvents,
    IWebEnvironmentState,
    IWebRenderState,
    RenderQuality
} from '../interfaces/game.interface';
import { calculateFPS, validateGameState, processEnvironmentData } from '../utils/game.utils';
import { LIDAR_PERFORMANCE, LIDAR_SCAN_SETTINGS } from '../constants/lidar.constants';

// Type definitions for context and actions
type GameAction = 
    | { type: 'START_SESSION'; payload: { sessionId: string; fleetConfig?: any } }
    | { type: 'UPDATE_ENVIRONMENT'; payload: IWebEnvironmentState }
    | { type: 'UPDATE_RENDER_STATE'; payload: Partial<IWebRenderState> }
    | { type: 'UPDATE_FLEET_STATE'; payload: any }
    | { type: 'END_SESSION' }
    | { type: 'SET_ERROR'; payload: string };

// Initial state with strict type safety
const INITIAL_GAME_STATE: IWebGameState = {
    gameId: '',
    sessionId: '',
    state: GameStates.INITIALIZING,
    environmentData: null,
    renderState: {
        resolution: {
            width: window.innerWidth,
            height: window.innerHeight
        },
        quality: RenderQuality.HIGH,
        lidarOverlayEnabled: true
    },
    fleetSync: {
        connected: false,
        deviceCount: 0,
        latency: 0,
        crdt: null
    },
    performance: {
        fps: 0,
        frameTime: 0,
        memoryUsage: 0
    }
};

// Enhanced reducer with CRDT support
function gameReducer(state: IWebGameState, action: GameAction): IWebGameState {
    switch (action.type) {
        case 'START_SESSION':
            return {
                ...state,
                sessionId: action.payload.sessionId,
                state: GameStates.LOADING,
                fleetSync: {
                    ...state.fleetSync,
                    connected: false,
                    deviceCount: 0
                }
            };

        case 'UPDATE_ENVIRONMENT': {
            const optimizedEnvironment = processEnvironmentData(
                action.payload,
                state.performance.fps
            );
            return {
                ...state,
                environmentData: optimizedEnvironment,
                state: GameStates.RUNNING
            };
        }

        case 'UPDATE_RENDER_STATE':
            return {
                ...state,
                renderState: {
                    ...state.renderState,
                    ...action.payload
                }
            };

        case 'UPDATE_FLEET_STATE':
            return {
                ...state,
                fleetSync: {
                    ...state.fleetSync,
                    ...action.payload
                }
            };

        case 'END_SESSION':
            return {
                ...INITIAL_GAME_STATE,
                state: GameStates.ENDED
            };

        case 'SET_ERROR':
            return {
                ...state,
                state: GameStates.ERROR
            };

        default:
            return state;
    }
}

// Create context with strict typing
const GameContext = createContext<{
    state: IWebGameState;
    dispatch: React.Dispatch<GameAction>;
} | undefined>(undefined);

// Enhanced provider with performance optimization
export function GameProvider({ children }: React.PropsWithChildren<{}>) {
    const [state, dispatch] = useReducer(gameReducer, INITIAL_GAME_STATE);
    const gameService = useMemo(() => new GameService(), []);
    const crdtDoc = useMemo(() => Automerge.init<any>(), []);

    // Performance monitoring setup
    useEffect(() => {
        let frameId: number;
        let lastFrameTime = performance.now();

        const updatePerformance = () => {
            const currentTime = performance.now();
            const { fps, shouldOptimize } = calculateFPS(
                currentTime,
                lastFrameTime,
                state.renderState
            );

            if (shouldOptimize) {
                dispatch({
                    type: 'UPDATE_RENDER_STATE',
                    payload: {
                        quality: fps < 45 ? RenderQuality.MEDIUM : RenderQuality.HIGH
                    }
                });
            }

            lastFrameTime = currentTime;
            frameId = requestAnimationFrame(updatePerformance);
        };

        frameId = requestAnimationFrame(updatePerformance);
        return () => cancelAnimationFrame(frameId);
    }, []);

    // CRDT synchronization
    useEffect(() => {
        if (state.state !== GameStates.RUNNING) return;

        const syncInterval = setInterval(() => {
            if (state.environmentData) {
                const changes = Automerge.getChanges(crdtDoc, Automerge.init());
                gameService.syncFleetState(changes, state.environmentData)
                    .catch(error => {
                        console.error('Fleet sync failed:', error);
                        dispatch({ type: 'SET_ERROR', payload: error.message });
                    });
            }
        }, LIDAR_SCAN_SETTINGS.PROCESSING_LATENCY_LIMIT);

        return () => clearInterval(syncInterval);
    }, [state.state, state.environmentData, crdtDoc, gameService]);

    // Session management
    const startSession = useCallback(async (sessionId: string, fleetConfig?: any) => {
        try {
            await gameService.startGameSession(sessionId, fleetConfig);
            dispatch({ type: 'START_SESSION', payload: { sessionId, fleetConfig } });
        } catch (error) {
            console.error('Failed to start session:', error);
            dispatch({ type: 'SET_ERROR', payload: error.message });
        }
    }, [gameService]);

    // Environment updates
    const updateEnvironment = useCallback((environmentData: IWebEnvironmentState) => {
        const { isValid, conflicts } = validateGameState(state, crdtDoc);
        if (!isValid) {
            console.error('State validation failed:', conflicts);
            return;
        }

        dispatch({ type: 'UPDATE_ENVIRONMENT', payload: environmentData });
    }, [state, crdtDoc]);

    const contextValue = useMemo(() => ({
        state,
        dispatch,
        startSession,
        updateEnvironment
    }), [state, dispatch, startSession, updateEnvironment]);

    return (
        <GameContext.Provider value={contextValue}>
            {children}
        </GameContext.Provider>
    );
}

// Custom hook for accessing game context
export function useGame() {
    const context = useContext(GameContext);
    if (context === undefined) {
        throw new Error('useGame must be used within a GameProvider');
    }
    return context;
}

export default GameContext;