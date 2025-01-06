import React, { useCallback, useEffect, useRef, useState } from 'react';
import styled from '@emotion/styled';
import Header from '../components/layout/Header';
import LidarOverlay from '../components/lidar/LidarOverlay';
import GameStats from '../components/game/GameStats';
import { useGame } from '../hooks/useGame';

// Styled components with GPU acceleration and HDR support
const LayoutContainer = styled.div`
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: var(--color-background);
  display: flex;
  flex-direction: column;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000;
  will-change: transform;
  
  /* HDR color support */
  color-gamut: p3;
  color-space: display-p3;
  
  /* Power-aware animations */
  transition: all var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1);
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const GameViewport = styled.main`
  flex: 1;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  
  /* Touch optimization */
  touch-action: none;
  -webkit-tap-highlight-color: transparent;
  
  /* GPU acceleration */
  transform: translateZ(0);
  will-change: transform, opacity;
  contain: content;
`;

const StatsContainer = styled.div`
  position: absolute;
  top: 80px;
  right: 24px;
  z-index: 100;
  pointer-events: none;
  
  /* GPU acceleration */
  transform: translateZ(0);
  will-change: transform;
  backdrop-filter: blur(8px);
`;

// Props interface with HDR and performance configuration
interface GameLayoutProps {
  children: React.ReactNode;
  className?: string;
  displayConfig?: {
    hdrEnabled: boolean;
    colorSpace: 'srgb' | 'display-p3';
  };
  performanceMode?: 'HIGH_PERFORMANCE' | 'BALANCED' | 'POWER_SAVER';
}

/**
 * High-performance game layout component with HDR support and performance optimization
 */
const GameLayout: React.FC<GameLayoutProps> = React.memo(({
  children,
  className,
  displayConfig = { hdrEnabled: true, colorSpace: 'display-p3' },
  performanceMode = 'BALANCED'
}) => {
  // Game state management
  const { gameState } = useGame();
  const { environmentData, fleetState } = gameState;

  // Refs for performance optimization
  const viewportRef = useRef<HTMLDivElement>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const [viewportDimensions, setViewportDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  // Handle viewport resize with debouncing
  const handleResize = useCallback(() => {
    if (!viewportRef.current) return;

    const { width, height } = viewportRef.current.getBoundingClientRect();
    setViewportDimensions({ width, height });
  }, []);

  // Initialize resize observer
  useEffect(() => {
    if (!viewportRef.current) return;

    resizeObserverRef.current = new ResizeObserver(handleResize);
    resizeObserverRef.current.observe(viewportRef.current);

    return () => {
      resizeObserverRef.current?.disconnect();
    };
  }, [handleResize]);

  // Configure power-aware animations
  useEffect(() => {
    document.documentElement.style.setProperty(
      '--animation-duration',
      performanceMode === 'POWER_SAVER' ? '500ms' : '300ms'
    );
  }, [performanceMode]);

  return (
    <LayoutContainer className={className}>
      <Header
        transparent
        powerMode={performanceMode}
        colorSpace={displayConfig.colorSpace}
      />

      <GameViewport ref={viewportRef}>
        {children}

        {/* LiDAR visualization overlay */}
        <LidarOverlay
          width={viewportDimensions.width}
          height={viewportDimensions.height}
          visualConfig={{
            pointSize: performanceMode === 'POWER_SAVER' ? 1 : 2,
            opacity: 0.8,
            quality: performanceMode === 'HIGH_PERFORMANCE' ? 'HIGH' : 'MEDIUM'
          }}
        />

        {/* Performance stats overlay */}
        <StatsContainer>
          <GameStats
            refreshRate={performanceMode === 'HIGH_PERFORMANCE' ? 16.67 : 33.33}
          />
        </StatsContainer>
      </GameViewport>
    </LayoutContainer>
  );
});

GameLayout.displayName = 'GameLayout';

export default GameLayout;