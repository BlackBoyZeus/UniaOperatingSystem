import React, { useCallback, useEffect, useMemo } from 'react';
import styled from '@emotion/styled';
import { ErrorBoundary } from 'react-error-boundary';
import usePerformanceMonitor from '@performance-monitor/react';

// Internal imports
import DashboardLayout from '../layouts/DashboardLayout';
import FleetStats from '../components/fleet/FleetStats';
import LidarStats from '../components/lidar/LidarStats';
import GameStats from '../components/game/GameStats';
import useAuth from '../hooks/useAuth';
import useFleet from '../hooks/useFleet';

// Types
type PowerMode = 'HIGH_PERFORMANCE' | 'BALANCED' | 'POWER_SAVER';

// Props interface
interface DashboardProps {
  className?: string;
  powerMode: PowerMode;
  hdrEnabled: boolean;
}

// GPU-accelerated styled components
const DashboardContainer = styled.div<{
  powerMode: PowerMode;
  hdrEnabled: boolean;
}>`
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform;
  contain: content;
  
  /* HDR color support */
  color-space: ${props => props.hdrEnabled ? 'display-p3' : 'srgb'};
  
  /* Power-aware animations */
  transition: all ${props => 
    props.powerMode === 'HIGH_PERFORMANCE' ? '150ms' :
    props.powerMode === 'POWER_SAVER' ? '500ms' : '300ms'
  } cubic-bezier(0.4, 0, 0.2, 1);
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const StatsGrid = styled.div<{ isCompact: boolean }>`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  width: 100%;
  
  /* Content containment for performance */
  contain: layout style paint;
  
  /* Responsive adjustments */
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
  
  /* Power-save mode adjustments */
  @media (prefers-reduced-power: high) {
    gap: 16px;
  }
`;

/**
 * Main dashboard page component with performance optimization and error handling
 */
const Dashboard: React.FC<DashboardProps> = React.memo(({
  className,
  powerMode = 'BALANCED',
  hdrEnabled = false
}) => {
  const { user, monitorSecurityEvents } = useAuth();
  const { currentFleet, networkStats } = useFleet();
  const { startMonitoring, metrics } = usePerformanceMonitor();

  // Initialize performance monitoring
  useEffect(() => {
    startMonitoring({
      maxLatency: 16, // Target 60fps
      sampleInterval: 1000,
      reportCallback: (metrics) => {
        if (metrics.averageLatency > 16) {
          console.warn('Performance degradation detected');
        }
      }
    });
  }, [startMonitoring]);

  // Security monitoring
  useEffect(() => {
    if (user) {
      const securityInterval = setInterval(() => {
        monitorSecurityEvents().catch(console.error);
      }, 30000);
      return () => clearInterval(securityInterval);
    }
  }, [user, monitorSecurityEvents]);

  // Memoized layout configuration
  const layoutConfig = useMemo(() => ({
    isCompact: window.innerWidth < 768,
    powerMode,
    hdrEnabled
  }), [powerMode, hdrEnabled]);

  return (
    <ErrorBoundary
      fallback={<div>Error loading dashboard</div>}
      onError={(error) => console.error('Dashboard Error:', error)}
    >
      <DashboardLayout>
        <DashboardContainer
          className={className}
          powerMode={powerMode}
          hdrEnabled={hdrEnabled}
        >
          <StatsGrid isCompact={layoutConfig.isCompact}>
            <FleetStats
              powerMode={powerMode}
              hdrEnabled={hdrEnabled}
            />
            
            <LidarStats
              powerMode={powerMode}
              colorMode={hdrEnabled ? 'HDR' : 'SDR'}
            />
            
            <GameStats
              refreshRate={powerMode === 'HIGH_PERFORMANCE' ? 16.67 : 33.33}
            />
          </StatsGrid>
        </DashboardContainer>
      </DashboardLayout>
    </ErrorBoundary>
  );
});

Dashboard.displayName = 'Dashboard';

export default Dashboard;