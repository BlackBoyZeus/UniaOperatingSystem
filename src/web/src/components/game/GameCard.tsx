import React, { useCallback, useEffect, useMemo } from 'react';
import styled from '@emotion/styled';
import { usePerformanceMonitor } from '@performance-monitor/react';
import * as Automerge from 'automerge';

// Internal imports
import { Card, CardProps } from '../common/Card';
import { 
    IWebGameState, 
    IWebEnvironmentState, 
    IWebRenderState, 
    IWebFleetState,
    GameStates 
} from '../../interfaces/game.interface';
import { GameService } from '../../services/game.service';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 * @external @performance-monitor/react v1.0.0
 * @external automerge v2.0.0
 */

interface GameCardProps extends Omit<CardProps, 'children'> {
    gameState: IWebGameState;
    onJoinGame: (gameId: string) => Promise<void>;
    onLeaveGame: () => Promise<void>;
    isActive: boolean;
    className?: string;
    fleetState: IWebFleetState;
    environmentState: IWebEnvironmentState;
    isHDREnabled?: boolean;
    powerMode?: number;
}

const StyledGameCard = styled(Card)<{
    isActive: boolean;
    isHDREnabled: boolean;
    powerMode: number;
}>`
    /* Base styles with GPU acceleration */
    position: relative;
    width: 100%;
    transform: var(--animation-gpu);
    will-change: transform, opacity;
    backface-visibility: hidden;
    contain: content;

    /* HDR-aware colors and effects */
    background: ${({ isHDREnabled }) => 
        isHDREnabled 
            ? 'color(display-p3 0.15 0.15 0.2)' 
            : 'var(--color-surface)'};
    
    border: 1px solid ${({ isActive, isHDREnabled }) =>
        isActive
            ? isHDREnabled
                ? 'color(display-p3 0.6 0.4 1)'
                : 'var(--color-primary)'
            : 'transparent'};

    /* Power-aware animations */
    transition: all ${({ powerMode }) =>
        powerMode > 0
            ? 'var(--animation-duration-power-save)'
            : 'var(--animation-duration)'};

    /* LiDAR overlay container */
    .lidar-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        mix-blend-mode: screen;
        opacity: ${({ powerMode }) => powerMode > 0 ? 0.6 : 0.8};
    }

    /* Fleet status indicators */
    .fleet-status {
        position: absolute;
        top: calc(var(--spacing-unit) * 1);
        right: calc(var(--spacing-unit) * 1);
        display: flex;
        align-items: center;
        gap: calc(var(--spacing-unit) * 0.5);
    }

    /* Performance optimized status dot */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: ${({ isActive, isHDREnabled }) =>
            isActive
                ? isHDREnabled
                    ? 'color(display-p3 0 1 0)'
                    : 'var(--color-secondary)'
                : 'var(--color-surface)'};
        transform: var(--animation-gpu);
    }
`;

export const GameCard: React.FC<GameCardProps> = ({
    gameState,
    onJoinGame,
    onLeaveGame,
    isActive,
    className,
    fleetState,
    environmentState,
    isHDREnabled = false,
    powerMode = 0,
    ...props
}) => {
    const { trackInteraction } = usePerformanceMonitor();
    const gameService = useMemo(() => new GameService(), []);

    // CRDT state management
    const crdtDoc = useMemo(() => Automerge.init<IWebGameState>(), []);
    const [syncedState, setSyncedState] = React.useState(crdtDoc);

    // Memoized game status for performance
    const gameStatus = useMemo(() => {
        return {
            isRunning: gameState.state === GameStates.RUNNING,
            hasEnvironmentData: !!environmentState,
            fleetSize: fleetState?.members?.length || 0
        };
    }, [gameState.state, environmentState, fleetState]);

    // Handle game join with performance tracking
    const handleJoinGame = useCallback(async (event: React.MouseEvent) => {
        event.preventDefault();
        const interactionId = trackInteraction('game_join');

        try {
            await onJoinGame(gameState.gameId);
            
            // Update CRDT state
            const newDoc = Automerge.change(syncedState, doc => {
                doc.state = GameStates.RUNNING;
                doc.sessionId = gameState.sessionId;
            });
            setSyncedState(newDoc);

            // Initialize LiDAR overlay if available
            if (environmentState) {
                await gameService.updateLiDAROverlay(environmentState);
            }

        } catch (error) {
            console.error('Failed to join game:', error);
        } finally {
            trackInteraction(interactionId, 'complete');
        }
    }, [gameState, onJoinGame, syncedState, environmentState, gameService, trackInteraction]);

    // Sync environment state with fleet
    useEffect(() => {
        if (gameStatus.isRunning && environmentState) {
            const syncInterval = setInterval(() => {
                gameService.syncFleetState(environmentState);
            }, 50); // 20Hz sync rate

            return () => clearInterval(syncInterval);
        }
    }, [gameStatus.isRunning, environmentState, gameService]);

    // Monitor and optimize performance
    useEffect(() => {
        const performanceInterval = setInterval(() => {
            if (gameStatus.isRunning) {
                const metrics = gameService.getMetrics();
                if (metrics.fps < 58) { // Below target 60 FPS
                    gameService.updateEnvironment({
                        ...environmentState,
                        quality: 'MEDIUM'
                    });
                }
            }
        }, 1000);

        return () => clearInterval(performanceInterval);
    }, [gameStatus.isRunning, environmentState, gameService]);

    return (
        <StyledGameCard
            isActive={isActive}
            isHDREnabled={isHDREnabled}
            powerMode={powerMode}
            onClick={!isActive ? handleJoinGame : undefined}
            className={className}
            variant="elevated"
            interactive={!isActive}
            {...props}
        >
            <div className="game-content">
                <h3>{gameState.gameId}</h3>
                <p>Session: {gameState.sessionId}</p>
                {environmentState && (
                    <div className="lidar-overlay">
                        {/* LiDAR visualization rendered here */}
                    </div>
                )}
            </div>
            
            <div className="fleet-status">
                <div className="status-dot" />
                <span>{gameStatus.fleetSize}/32 Players</span>
            </div>
        </StyledGameCard>
    );
};

export type { GameCardProps };