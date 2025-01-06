import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { useWebGL } from '@react-three/fiber';

// Internal imports
import { GameList, GameListProps } from '../components/game/GameList';
import { LidarOverlay } from '../components/lidar/LidarOverlay';
import { useGame } from '../hooks/useGame';
import { useFleet } from '../hooks/useFleet';

// Constants for performance optimization
const PERFORMANCE_THRESHOLDS = {
    FPS: 58,
    MEMORY: 0.8,
    LATENCY: 50
} as const;

const LIDAR_CONFIG = {
    resolution: 0.01,
    scanFrequency: 30,
    range: 5.0
} as const;

interface GamePageProps {
    className?: string;
    lidarConfig?: {
        resolution: number;
        scanFrequency: number;
        range: number;
    };
    fleetConfig?: {
        maxSize: number;
        autoJoin: boolean;
        syncInterval: number;
    };
    performanceConfig?: {
        minFps: number;
        maxLatency: number;
        memoryLimit: number;
    };
}

const StyledGamePage = styled.div<{ isActive: boolean }>`
    /* Base styles with GPU acceleration */
    position: relative;
    width: 100%;
    height: 100%;
    transform: translateZ(0);
    will-change: transform;
    contain: content;

    /* HDR-aware colors */
    background: ${({ theme }) => theme.colors.surface};
    @media (dynamic-range: high) {
        background: color(display-p3 0.15 0.15 0.2);
    }

    /* Game container with optimized rendering */
    .game-container {
        position: relative;
        width: 100%;
        height: 100%;
        transform: translateZ(0);
        will-change: transform;
        contain: layout style paint;
    }

    /* LiDAR overlay container */
    .lidar-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        mix-blend-mode: screen;
        opacity: ${({ isActive }) => isActive ? 0.8 : 0.6};
        transition: opacity 200ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Fleet status indicators */
    .fleet-status {
        position: absolute;
        top: 1rem;
        right: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 0.5rem;
        backdrop-filter: blur(4px);
    }

    /* Performance metrics display */
    .performance-metrics {
        position: absolute;
        bottom: 1rem;
        left: 1rem;
        padding: 0.5rem;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 0.5rem;
        backdrop-filter: blur(4px);
    }
`;

export const GamePage: React.FC<GamePageProps> = ({
    className,
    lidarConfig = LIDAR_CONFIG,
    fleetConfig = { maxSize: 32, autoJoin: true, syncInterval: 50 },
    performanceConfig = PERFORMANCE_THRESHOLDS
}) => {
    // Initialize hooks and context
    const { gl, contextLost, contextRestored } = useWebGL();
    const { gameState, startGame, endGame, updateEnvironment } = useGame();
    const { currentFleet, joinFleet, leaveFleet, networkStats } = useFleet();

    // Local state management
    const [isActive, setIsActive] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Memoized configuration
    const config = useMemo(() => ({
        lidar: {
            ...lidarConfig,
            quality: gameState.fps >= performanceConfig.minFps ? 'HIGH' : 'MEDIUM'
        },
        fleet: {
            ...fleetConfig,
            currentSize: currentFleet?.members.size || 0
        },
        performance: {
            ...performanceConfig,
            currentFps: gameState.fps
        }
    }), [gameState.fps, currentFleet?.members.size, lidarConfig, fleetConfig, performanceConfig]);

    // Handle game selection with fleet coordination
    const handleGameSelect = useCallback(async (gameId: string) => {
        try {
            setIsActive(true);
            setError(null);

            // Start game session
            await startGame();

            // Join or create fleet
            if (!currentFleet) {
                await joinFleet(gameId);
            }

            // Initialize LiDAR scanning
            if (gameState.environmentData) {
                await updateEnvironment(gameState.environmentData);
            }

        } catch (error) {
            console.error('Game initialization failed:', error);
            setError(error instanceof Error ? error : new Error('Game initialization failed'));
            setIsActive(false);
        }
    }, [startGame, joinFleet, updateEnvironment, currentFleet, gameState.environmentData]);

    // Handle WebGL context loss
    useEffect(() => {
        if (contextLost) {
            console.error('WebGL context lost, attempting recovery...');
            setError(new Error('Graphics context lost. Attempting recovery...'));
        }
    }, [contextLost]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (isActive) {
                endGame().catch(console.error);
                leaveFleet().catch(console.error);
            }
        };
    }, [isActive, endGame, leaveFleet]);

    return (
        <StyledGamePage className={className} isActive={isActive}>
            <div className="game-container">
                <GameList
                    onGameSelect={handleGameSelect}
                    isLoading={!isActive}
                    lidarConfig={config.lidar}
                    fleetOptions={config.fleet}
                    performanceTargets={config.performance}
                />

                {isActive && gameState.environmentData && (
                    <div className="lidar-overlay">
                        <LidarOverlay
                            width={window.innerWidth}
                            height={window.innerHeight}
                            visualConfig={config.lidar}
                        />
                    </div>
                )}

                {currentFleet && (
                    <div className="fleet-status">
                        <span>Fleet: {currentFleet.members.size}/32</span>
                        <span>Latency: {networkStats.averageLatency.toFixed(0)}ms</span>
                    </div>
                )}

                <div className="performance-metrics">
                    <div>FPS: {gameState.fps}</div>
                    <div>Quality: {config.lidar.quality}</div>
                </div>

                {error && (
                    <div className="error-message">
                        {error.message}
                    </div>
                )}
            </div>
        </StyledGamePage>
    );
};

export default GamePage;