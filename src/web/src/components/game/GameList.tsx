import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import styled from '@emotion/styled';
import { AutoSizer, List, WindowScroller } from 'react-virtualized';
import { useIntersectionObserver } from 'react-intersection-observer';

// Internal imports
import { GameCard, GameCardProps } from './GameCard';
import { useGame } from '../../hooks/useGame';
import { GameService } from '../../services/game.service';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 * @external react-virtualized v9.22.3
 * @external react-intersection-observer v9.5.2
 */

// Constants for performance optimization
const VIRTUALIZATION_OVERSCAN = 5;
const CARD_HEIGHT = 200;
const SCROLL_THROTTLE = 150;
const FLEET_UPDATE_INTERVAL = 50;
const PERFORMANCE_THRESHOLD = {
    FPS: 58,
    MEMORY: 0.8,
    LATENCY: 50
};

// Enhanced interfaces for component props
interface GameListProps {
    className?: string;
    onGameSelect: (gameId: string) => Promise<void>;
    isLoading?: boolean;
    lidarConfig: {
        resolution: number;
        scanFrequency: number;
        range: number;
    };
    fleetOptions: {
        maxSize: number;
        autoJoin: boolean;
        syncInterval: number;
    };
    performanceTargets: {
        minFps: number;
        maxLatency: number;
        memoryLimit: number;
    };
}

// GPU-accelerated styled components
const StyledGameList = styled.div<{ isLoading: boolean }>`
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

    /* Loading state with optimized blur */
    ${({ isLoading }) => isLoading && `
        opacity: 0.7;
        pointer-events: none;
        backdrop-filter: blur(2px);
        transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
    `}

    /* Performance optimizations */
    .virtualized-list {
        contain: strict;
        will-change: transform;
        backface-visibility: hidden;
    }

    .game-card {
        transform: translateZ(0);
        transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1);
        will-change: transform;
    }
`;

export const GameList: React.FC<GameListProps> = ({
    className,
    onGameSelect,
    isLoading = false,
    lidarConfig,
    fleetOptions,
    performanceTargets
}) => {
    // State and refs
    const [games, setGames] = useState<GameCardProps[]>([]);
    const [activeGameId, setActiveGameId] = useState<string | null>(null);
    const listRef = useRef<List>(null);
    const gameService = useRef(new GameService());

    // Custom hooks
    const { gameState, updateEnvironment, syncFleetState } = useGame();
    const { ref: scrollRef, inView } = useIntersectionObserver({
        threshold: 0.1,
        rootMargin: '100px'
    });

    // Memoized list configuration
    const listConfig = useMemo(() => ({
        overscanRowCount: VIRTUALIZATION_OVERSCAN,
        rowHeight: CARD_HEIGHT,
        rowCount: games.length,
        threshold: PERFORMANCE_THRESHOLD
    }), [games.length]);

    // Initialize game service and CRDT sync
    useEffect(() => {
        const service = gameService.current;
        const syncInterval = setInterval(() => {
            if (activeGameId) {
                service.syncFleetState(activeGameId)
                    .catch(error => console.error('Fleet sync failed:', error));
            }
        }, FLEET_UPDATE_INTERVAL);

        return () => {
            clearInterval(syncInterval);
            service.dispose();
        };
    }, [activeGameId]);

    // Handle game selection with performance tracking
    const handleGameSelect = useCallback(async (gameId: string) => {
        try {
            setActiveGameId(gameId);
            await onGameSelect(gameId);

            // Initialize LiDAR scanning
            await gameService.current.startGameSession(gameId, {
                lidarConfig,
                fleetOptions,
                performanceTargets
            });
        } catch (error) {
            console.error('Game selection failed:', error);
            setActiveGameId(null);
        }
    }, [onGameSelect, lidarConfig, fleetOptions, performanceTargets]);

    // Optimized row renderer for virtualized list
    const rowRenderer = useCallback(({
        index,
        key,
        style
    }: {
        index: number;
        key: string;
        style: React.CSSProperties;
    }) => {
        const game = games[index];
        return (
            <div key={key} style={style} className="game-card">
                <GameCard
                    gameState={game.gameState}
                    onJoinGame={() => handleGameSelect(game.gameState.gameId)}
                    isActive={game.gameState.gameId === activeGameId}
                    fleetState={game.fleetState}
                    environmentState={game.environmentState}
                    isHDREnabled={window.matchMedia('(dynamic-range: high)').matches}
                    powerMode={navigator.powerSaveMode ? 1 : 0}
                />
            </div>
        );
    }, [games, activeGameId, handleGameSelect]);

    // Performance-optimized scroll handler
    const handleScroll = useCallback(({ scrollTop }: { scrollTop: number }) => {
        if (listRef.current) {
            listRef.current.scrollTop = scrollTop;
        }
    }, []);

    return (
        <StyledGameList className={className} isLoading={isLoading} ref={scrollRef}>
            <WindowScroller onScroll={handleScroll}>
                {({ height, isScrolling, registerChild, scrollTop }) => (
                    <AutoSizer disableHeight>
                        {({ width }) => (
                            <div ref={registerChild}>
                                <List
                                    ref={listRef}
                                    autoHeight
                                    height={height}
                                    width={width}
                                    isScrolling={isScrolling}
                                    scrollTop={scrollTop}
                                    rowRenderer={rowRenderer}
                                    {...listConfig}
                                    className="virtualized-list"
                                />
                            </div>
                        )}
                    </AutoSizer>
                )}
            </WindowScroller>
        </StyledGameList>
    );
};

// Export types for external usage
export type { GameListProps };