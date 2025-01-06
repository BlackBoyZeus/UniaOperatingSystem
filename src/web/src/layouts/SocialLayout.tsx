import React, { useCallback, useEffect, useMemo } from 'react';
import styled from '@emotion/styled';
import { useParams } from 'react-router-dom';

import DashboardLayout, { DashboardLayoutProps } from './DashboardLayout';
import SocialHub from '../components/social/SocialHub';
import { UI_CONSTANTS } from '../constants/ui.constants';

// Types for power and network quality management
type PowerMode = 'HIGH_PERFORMANCE' | 'BALANCED' | 'POWER_SAVER';
type NetworkQuality = 'excellent' | 'good' | 'fair' | 'poor';

// Props interface for the SocialLayout component
interface SocialLayoutProps extends Omit<DashboardLayoutProps, 'children'> {
  children?: React.ReactNode;
  className?: string;
  powerMode: PowerMode;
  hdrEnabled: boolean;
  fleetSize: number;
  networkQuality: NetworkQuality;
}

// GPU-accelerated styled container with HDR support
const SocialContainer = styled.div<{
  powerMode: PowerMode;
  hdrEnabled: boolean;
}>`
  display: flex;
  flex-direction: column;
  flex: 1;
  gap: ${UI_CONSTANTS.SPACING.CONTAINER_GAP}px;
  padding: ${UI_CONSTANTS.SPACING.CONTAINER_PADDING}px;
  
  /* GPU acceleration optimizations */
  will-change: transform, opacity;
  transform: translate3d(0, 0, 0);
  backface-visibility: hidden;
  
  /* Power-aware animations */
  transition: all ${props => 
    props.powerMode === 'POWER_SAVER' 
      ? UI_CONSTANTS.ANIMATION.DURATION.POWER_SAVER 
      : props.powerMode === 'HIGH_PERFORMANCE'
        ? UI_CONSTANTS.ANIMATION.DURATION.FAST
        : UI_CONSTANTS.ANIMATION.DURATION.NORMAL}ms ${UI_CONSTANTS.ANIMATION.EASING.DEFAULT};
  
  /* HDR color management */
  color-gamut: ${props => props.hdrEnabled ? 'p3' : 'srgb'};
  background-color: ${props => props.theme.colors.background};

  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

// Enhanced SocialLayout component with hardware security and performance optimizations
const SocialLayout: React.FC<SocialLayoutProps> = React.memo(({
  className,
  powerMode,
  hdrEnabled,
  fleetSize,
  networkQuality,
  ...props
}) => {
  // Extract fleet ID from route parameters
  const { fleetId } = useParams<{ fleetId: string }>();

  // Memoized performance configuration
  const performanceConfig = useMemo(() => ({
    powerMode,
    hdrEnabled,
    maxFleetSize: 32,
    networkQualityThreshold: networkQuality === 'excellent' ? 0.9 :
                            networkQuality === 'good' ? 0.7 :
                            networkQuality === 'fair' ? 0.5 : 0.3
  }), [powerMode, hdrEnabled, networkQuality]);

  // Performance monitoring callback
  const handlePerformanceIssue = useCallback((metric: string, value: number) => {
    console.warn(`Performance issue detected: ${metric} = ${value}`);
  }, []);

  // Setup performance monitoring
  useEffect(() => {
    const performanceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > UI_CONSTANTS.ANIMATION.FRAME_BUDGET) {
          handlePerformanceIssue('frame_time', entry.duration);
        }
      }
    });

    performanceObserver.observe({ entryTypes: ['frame'] });
    return () => performanceObserver.disconnect();
  }, [handlePerformanceIssue]);

  return (
    <DashboardLayout {...props}>
      <SocialContainer
        className={className}
        powerMode={powerMode}
        hdrEnabled={hdrEnabled}
        data-testid="social-layout"
      >
        <SocialHub
          fleetId={fleetId}
          securityContext="social"
          privacySettings={{
            shareLocation: true,
            shareScanData: true,
            dataRetentionDays: 30,
            gdprConsent: true
          }}
          region="auto"
          powerMode={powerMode === 'POWER_SAVER' ? 'powersave' :
                     powerMode === 'HIGH_PERFORMANCE' ? 'performance' :
                     'balanced'}
        />
      </SocialContainer>
    </DashboardLayout>
  );
});

// Display name for debugging
SocialLayout.displayName = 'SocialLayout';

export type { SocialLayoutProps };
export default SocialLayout;