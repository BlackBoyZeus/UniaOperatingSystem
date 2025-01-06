/**
 * @file GameStats.tsx
 * @version 1.0.0
 * @description High-performance React component for displaying real-time game statistics
 * with GPU acceleration and performance optimization for TALD UNIA platform
 */

import React, { useMemo } from 'react';
import styled from '@emotion/styled';
import throttle from 'lodash/throttle';

// Internal imports
import { useGame } from '../../hooks/useGame';
import Progress from '../common/Progress';
import { THEME_SETTINGS, PERFORMANCE_THRESHOLDS } from '../../constants/ui.constants';

// Performance thresholds based on technical specifications
const PERFORMANCE_METRICS = {
  FPS: {
    warning: 45,
    critical: 30,
    target: 60
  },
  SCAN_QUALITY: {
    warning: 70,
    critical: 50,
    target: 95
  },
  LATENCY: {
    warning: 35,
    critical: 45,
    target: 50
  }
} as const;

// Styled components with GPU acceleration
const StatsContainer = styled.div`
  position: relative;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  padding: 16px;
  background: rgba(18, 18, 18, 0.85);
  border-radius: 8px;
  backdrop-filter: blur(8px);
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  will-change: transform;
  backface-visibility: hidden;
`;

const StatItem = styled.div<{ isWarning?: boolean; isCritical?: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 8px;
  color: ${({ isWarning, isCritical }) => 
    isCritical ? THEME_SETTINGS.DARK.accent :
    isWarning ? THEME_SETTINGS.DARK.hdrColors.secondary :
    THEME_SETTINGS.DARK.contrast.high};
  
  /* Text rendering optimization */
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
`;

const Label = styled.span`
  font-size: 14px;
  font-weight: 500;
  opacity: 0.9;
`;

interface GameStatsProps {
  className?: string;
  refreshRate?: number;
}

/**
 * Calculates scan quality percentage based on multiple metrics
 */
const calculateScanQuality = throttle((environmentData: any) => {
  if (!environmentData) return 0;

  const pointCloudDensity = environmentData.pointCloud?.length || 0;
  const maxPoints = 1_200_000; // From technical specifications
  const densityScore = Math.min((pointCloudDensity / maxPoints) * 100, 100);

  return Math.round(densityScore);
}, 100);

/**
 * Determines color based on performance thresholds
 */
const getPerformanceColor = (value: number, thresholds: { warning: number; critical: number }) => {
  if (value <= thresholds.critical) return 'accent';
  if (value <= thresholds.warning) return 'secondary';
  return 'primary';
};

/**
 * GameStats component for displaying real-time performance metrics
 */
const GameStats: React.FC<GameStatsProps> = React.memo(({ className, refreshRate = 16.67 }) => {
  const { gameState } = useGame();
  const { environmentData, fps, networkLatency } = gameState;

  // Memoized calculations for performance metrics
  const metrics = useMemo(() => {
    const scanQuality = calculateScanQuality(environmentData);
    const fpsPercentage = (fps / PERFORMANCE_METRICS.FPS.target) * 100;
    const latencyPercentage = ((PERFORMANCE_METRICS.LATENCY.target - networkLatency) / 
      PERFORMANCE_METRICS.LATENCY.target) * 100;

    return {
      fps: {
        value: Math.min(fpsPercentage, 100),
        color: getPerformanceColor(fps, PERFORMANCE_METRICS.FPS)
      },
      scanQuality: {
        value: scanQuality,
        color: getPerformanceColor(scanQuality, PERFORMANCE_METRICS.SCAN_QUALITY)
      },
      latency: {
        value: Math.max(latencyPercentage, 0),
        color: getPerformanceColor(networkLatency, PERFORMANCE_METRICS.LATENCY)
      }
    };
  }, [environmentData, fps, networkLatency]);

  return (
    <StatsContainer className={className}>
      <StatItem>
        <Label>FPS</Label>
        <Progress
          value={metrics.fps.value}
          color={metrics.fps.color}
          size="small"
          powerMode="performance"
        />
      </StatItem>

      <StatItem>
        <Label>Scan Quality</Label>
        <Progress
          value={metrics.scanQuality.value}
          color={metrics.scanQuality.color}
          size="small"
          powerMode="performance"
        />
      </StatItem>

      <StatItem>
        <Label>Network Latency</Label>
        <Progress
          value={metrics.latency.value}
          color={metrics.latency.color}
          size="small"
          powerMode="performance"
        />
      </StatItem>

      <StatItem>
        <Label>Environment Data</Label>
        <Progress
          value={environmentData ? 100 : 0}
          color="primary"
          size="small"
          powerMode="performance"
        />
      </StatItem>
    </StatsContainer>
  );
});

GameStats.displayName = 'GameStats';

export default GameStats;