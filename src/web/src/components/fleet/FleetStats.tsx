/**
 * @file FleetStats.tsx
 * @version 1.0.0
 * @description Real-time fleet statistics component with HDR support and power-aware optimizations
 */

import React, { useMemo, useCallback } from 'react';
import styled from '@emotion/styled';
import { useFleet } from '../../hooks/useFleet';
import { Progress } from '../common/Progress';
import { 
    THEME_SETTINGS, 
    ANIMATION_TIMINGS, 
    PERFORMANCE_THRESHOLDS 
} from '../../constants/ui.constants';

// Props interface with HDR and power optimization support
interface FleetStatsProps {
    className?: string;
    hdrEnabled?: boolean;
    powerMode?: 'performance' | 'balanced' | 'powersave';
}

// GPU-accelerated styled components with HDR support
const StatsContainer = styled.div<{ hdrEnabled?: boolean }>`
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    padding: 16px;
    background: ${({ hdrEnabled }) => 
        hdrEnabled ? 'color(display-p3 0.07 0.07 0.07)' : THEME_SETTINGS.DARK.background};
    border-radius: 8px;
    transform: translateZ(0);
    will-change: transform;
    backface-visibility: hidden;
`;

const StatItem = styled.div<{ isWarning?: boolean; hdrEnabled?: boolean }>`
    display: flex;
    flex-direction: column;
    gap: 8px;
    color: ${({ isWarning, hdrEnabled }) => 
        isWarning 
            ? (hdrEnabled ? 'color(display-p3 1 0.3 0.2)' : THEME_SETTINGS.DARK.accent)
            : THEME_SETTINGS.DARK.text};
    transition: color ${ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
`;

const Label = styled.span`
    font-size: 14px;
    opacity: 0.87;
    text-rendering: optimizeLegibility;
`;

const Value = styled.span`
    font-size: 24px;
    font-weight: 500;
    letter-spacing: 0.5px;
`;

/**
 * Enhanced fleet statistics component with real-time monitoring
 */
const FleetStats: React.FC<FleetStatsProps> = React.memo(({
    className,
    hdrEnabled = false,
    powerMode = 'balanced'
}) => {
    const { 
        currentFleet, 
        networkStats, 
        qualityMetrics 
    } = useFleet();

    // Memoized fleet capacity calculation
    const capacityStats = useMemo(() => {
        if (!currentFleet) return { used: 0, total: 32, percentage: 0 };
        const used = currentFleet.members.length;
        const total = currentFleet.maxDevices;
        return {
            used,
            total,
            percentage: (used / total) * 100
        };
    }, [currentFleet]);

    // Memoized network quality calculation with regional optimization
    const networkQuality = useMemo(() => {
        if (!networkStats || !qualityMetrics) return { score: 0, isWarning: false };
        
        const latencyScore = 1 - (networkStats.averageLatency / 50); // 50ms threshold
        const syncScore = qualityMetrics.syncSuccess / 100;
        const reliabilityScore = 1 - (networkStats.packetsLost / 100);
        
        const score = (latencyScore + syncScore + reliabilityScore) / 3 * 100;
        return {
            score: Math.round(score),
            isWarning: score < 80
        };
    }, [networkStats, qualityMetrics]);

    // Power-aware animation configuration
    const animationConfig = useMemo(() => ({
        duration: ANIMATION_TIMINGS.POWER_MODES[powerMode.toUpperCase()].transitionMultiplier * 
                 ANIMATION_TIMINGS.TRANSITION_DURATION,
        fps: ANIMATION_TIMINGS.POWER_MODES[powerMode.toUpperCase()].fps
    }), [powerMode]);

    // Optimized render for fleet statistics
    return (
        <StatsContainer 
            className={className}
            hdrEnabled={hdrEnabled}
            role="region"
            aria-label="Fleet Statistics"
        >
            <StatItem hdrEnabled={hdrEnabled}>
                <Label>Fleet Capacity</Label>
                <Value>{capacityStats.used} / {capacityStats.total}</Value>
                <Progress
                    value={capacityStats.percentage}
                    color="primary"
                    size="medium"
                    powerMode={powerMode}
                    hdrEnabled={hdrEnabled}
                />
            </StatItem>

            <StatItem 
                isWarning={networkQuality.isWarning}
                hdrEnabled={hdrEnabled}
            >
                <Label>Network Quality</Label>
                <Value>{networkQuality.score}%</Value>
                <Progress
                    value={networkQuality.score}
                    color={networkQuality.isWarning ? "accent" : "secondary"}
                    size="medium"
                    powerMode={powerMode}
                    hdrEnabled={hdrEnabled}
                />
            </StatItem>

            <StatItem hdrEnabled={hdrEnabled}>
                <Label>Average Latency</Label>
                <Value>
                    {networkStats?.averageLatency.toFixed(1)}ms
                </Value>
                <Progress
                    value={(networkStats?.averageLatency / 50) * 100}
                    color={networkStats?.averageLatency > 40 ? "accent" : "primary"}
                    size="medium"
                    powerMode={powerMode}
                    hdrEnabled={hdrEnabled}
                />
            </StatItem>

            <StatItem hdrEnabled={hdrEnabled}>
                <Label>Sync Success Rate</Label>
                <Value>
                    {qualityMetrics?.syncSuccess.toFixed(1)}%
                </Value>
                <Progress
                    value={qualityMetrics?.syncSuccess || 0}
                    color={qualityMetrics?.syncSuccess < 90 ? "accent" : "secondary"}
                    size="medium"
                    powerMode={powerMode}
                    hdrEnabled={hdrEnabled}
                />
            </StatItem>
        </StatsContainer>
    );
});

FleetStats.displayName = 'FleetStats';

export default FleetStats;