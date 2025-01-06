/**
 * @file Progress.tsx
 * @version 1.0.0
 * @description High-performance, GPU-accelerated progress indicator component with
 * HDR color support and power-aware optimizations for TALD UNIA platform
 */

import React from 'react';
import styled from '@emotion/styled';
import { keyframes } from '@emotion/react';
import { ANIMATION_TIMINGS, THEME_SETTINGS, POWER_MODES } from '../../constants/ui.constants';

// Progress component props interface with power and performance features
interface ProgressProps {
  value: number;
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'secondary' | 'accent';
  powerMode?: 'performance' | 'balanced' | 'powersave';
  showLabel?: boolean;
  className?: string;
}

// GPU-accelerated progress animation with power-aware timing
const progressAnimation = (powerMode: string) => keyframes`
  from {
    transform: translateX(-100%) translateZ(0);
  }
  to {
    transform: translateX(0) translateZ(0);
  }
`;

// Size variants mapping with GPU-optimized properties
const sizeMap = {
  small: { height: '4px', fontSize: '12px' },
  medium: { height: '8px', fontSize: '14px' },
  large: { height: '12px', fontSize: '16px' }
};

// GPU-accelerated container with power-aware rendering
const ProgressContainer = styled.div<{
  size: string;
  powerMode: string;
}>`
  position: relative;
  width: 100%;
  height: ${props => sizeMap[props.size as keyof typeof sizeMap].height};
  background-color: rgba(255, 255, 255, 0.12);
  border-radius: ${props => sizeMap[props.size as keyof typeof sizeMap].height};
  overflow: hidden;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000px;
  will-change: transform;
`;

// HDR-aware progress bar with GPU-accelerated animations
const ProgressBar = styled.div<{
  value: number;
  color: string;
  powerMode: string;
}>`
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: ${props => props.value}%;
  background-color: ${props => 
    THEME_SETTINGS.DARK.hdrColors[props.color as keyof typeof THEME_SETTINGS.DARK.hdrColors]};
  border-radius: inherit;
  
  /* Power-aware animation configuration */
  animation: ${props => progressAnimation(props.powerMode)}
    ${props => ANIMATION_TIMINGS.POWER_MODES[props.powerMode.toUpperCase()].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  will-change: transform;
`;

// Label container with performance optimizations
const ProgressLabel = styled.div<{ size: string }>`
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  font-size: ${props => sizeMap[props.size as keyof typeof sizeMap].fontSize};
  color: ${THEME_SETTINGS.DARK.contrast.high};
  
  /* GPU acceleration for text rendering */
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
`;

/**
 * High-performance Progress component with GPU acceleration and power optimization
 */
const Progress: React.FC<ProgressProps> = React.memo(({
  value,
  size = 'medium',
  color = 'primary',
  powerMode = 'balanced',
  showLabel = true,
  className
}) => {
  // Clamp value between 0 and 100
  const clampedValue = Math.min(Math.max(value, 0), 100);

  return (
    <ProgressContainer
      size={size}
      powerMode={powerMode}
      className={className}
      role="progressbar"
      aria-valuenow={clampedValue}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <ProgressBar
        value={clampedValue}
        color={color}
        powerMode={powerMode}
      />
      {showLabel && (
        <ProgressLabel size={size}>
          {Math.round(clampedValue)}%
        </ProgressLabel>
      )}
    </ProgressContainer>
  );
});

Progress.displayName = 'Progress';

export default Progress;