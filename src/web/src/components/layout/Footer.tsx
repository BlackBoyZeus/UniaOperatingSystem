import React from 'react';
import styled from 'styled-components';
import Icon from '../common/Icon';
import { ANIMATION_TIMINGS, LAYOUT_CONSTANTS, TYPOGRAPHY } from '../../constants/ui.constants';

/**
 * Power mode type for animation optimization
 */
type PowerMode = 'HIGH_PERFORMANCE' | 'BALANCED' | 'POWER_SAVER';

/**
 * Props interface for the enhanced Footer component
 */
interface FooterProps {
  batteryLevel: number;
  networkLatency: number;
  className?: string;
  powerMode?: PowerMode;
  hdrEnabled?: boolean;
}

/**
 * Styled footer component with GPU acceleration and HDR support
 */
const StyledFooter = styled.footer<{ powerMode?: PowerMode; hdrEnabled?: boolean }>`
  height: 40px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 ${LAYOUT_CONSTANTS.GRID_BASE * 2}px;
  background: ${props => props.theme.background};
  border-top: 1px solid ${props => props.hdrEnabled 
    ? props.theme.hdrColors.primary 
    : props.theme.primary};
  
  /* GPU acceleration optimizations */
  contain: layout style paint;
  content-visibility: auto;
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;
  
  /* Power-aware transitions */
  transition: all ${props => 
    props.powerMode === 'POWER_SAVER' 
      ? ANIMATION_TIMINGS.POWER_MODES.POWER_SAVER.transitionMultiplier * ANIMATION_TIMINGS.TRANSITION_DURATION
      : ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

/**
 * Styled metric container with HDR color support
 */
const MetricContainer = styled.div<{ hdrEnabled?: boolean }>`
  display: flex;
  align-items: center;
  gap: ${LAYOUT_CONSTANTS.GRID_BASE}px;
  color: ${props => props.hdrEnabled 
    ? props.theme.hdrColors.secondary 
    : props.theme.secondary};
  font-family: ${TYPOGRAPHY.FONT_FAMILY};
  font-size: ${TYPOGRAPHY.BASE_SIZE}px;
  transform: translateZ(0);
  
  /* Optimized transitions */
  transition: color ${ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

/**
 * Styled status indicator with power-aware animations
 */
const StatusIndicator = styled.div<{ powerMode?: PowerMode }>`
  display: flex;
  align-items: center;
  gap: ${LAYOUT_CONSTANTS.GRID_BASE / 2}px;
  font-family: ${TYPOGRAPHY.FONT_FAMILY};
  font-size: ${TYPOGRAPHY.BASE_SIZE}px;
  line-height: ${TYPOGRAPHY.LINE_HEIGHT};
  transform: translateZ(0);
  
  /* Power-aware animations */
  transition: opacity ${props => 
    props.powerMode === 'POWER_SAVER'
      ? ANIMATION_TIMINGS.POWER_MODES.POWER_SAVER.transitionMultiplier * ANIMATION_TIMINGS.TRANSITION_DURATION
      : ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
`;

/**
 * Enhanced footer component with GPU acceleration and power-aware optimizations
 */
const Footer: React.FC<FooterProps> = React.memo(({
  batteryLevel,
  networkLatency,
  className,
  powerMode = 'BALANCED',
  hdrEnabled = false
}) => {
  // Format battery level with power-aware color mapping
  const batteryColor = React.useMemo(() => {
    if (batteryLevel <= 20) {
      return hdrEnabled ? 'color(display-p3 1 0 0)' : '#FF0000';
    }
    return hdrEnabled ? 'color(display-p3 0 1 0)' : '#00FF00';
  }, [batteryLevel, hdrEnabled]);

  // Format network latency with performance indicators
  const networkStatus = React.useMemo(() => {
    if (networkLatency <= 50) return 'Excellent';
    if (networkLatency <= 100) return 'Good';
    return 'Poor';
  }, [networkLatency]);

  return (
    <StyledFooter 
      className={className}
      powerMode={powerMode}
      hdrEnabled={hdrEnabled}
    >
      <MetricContainer hdrEnabled={hdrEnabled}>
        <StatusIndicator powerMode={powerMode}>
          <Icon 
            name="battery"
            size={16}
            color={batteryColor}
            powerMode={powerMode}
          />
          {batteryLevel}%
        </StatusIndicator>
      </MetricContainer>

      <MetricContainer hdrEnabled={hdrEnabled}>
        <StatusIndicator powerMode={powerMode}>
          <Icon 
            name="network"
            size={16}
            animate={networkLatency > 100}
            powerMode={powerMode}
          />
          {networkLatency}ms ({networkStatus})
        </StatusIndicator>
      </MetricContainer>
    </StyledFooter>
  );
});

Footer.displayName = 'Footer';

export type { FooterProps };
export default Footer;