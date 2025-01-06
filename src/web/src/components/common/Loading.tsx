/**
 * @file Loading Component
 * @version 1.0.0
 * @description A high-performance loading component with GPU acceleration, 
 * power-aware animations, and HDR color support for the TALD UNIA platform.
 */

import React from 'react'; // v18.2.0
import styled from '@emotion/styled'; // v11.11.0
import { keyframes } from '@emotion/react'; // v11.11.0
import { ANIMATION_TIMINGS, THEME_SETTINGS } from '../../constants/ui.constants';

// Props interface with power and HDR awareness
interface LoadingProps {
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'secondary' | 'accent';
  overlay?: boolean;
  text?: string;
  powerMode?: 'high' | 'balanced' | 'low';
  hdrEnabled?: boolean;
  className?: string;
}

// GPU-accelerated spinner animation with power-aware timing
const spinAnimation = (powerMode: string = 'balanced') => keyframes`
  from {
    transform: translate3d(0, 0, 0) rotate(0deg);
    will-change: transform;
  }
  to {
    transform: translate3d(0, 0, 0) rotate(360deg);
    will-change: transform;
  }
`;

// Size mappings for different loading variants
const SIZE_MAP = {
  small: '24px',
  medium: '48px',
  large: '64px'
};

// Styled container with HDR background support
const LoadingContainer = styled.div<{ overlay?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  ${({ overlay }) => overlay && `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: ${THEME_SETTINGS.DARK.background};
    z-index: 100;
  `}
`;

// GPU-accelerated spinner with power-aware animation
const LoadingSpinner = styled.div<{
  size: string;
  color: string;
  powerMode: string;
  hdrEnabled: boolean;
}>`
  width: ${props => SIZE_MAP[props.size]};
  height: ${props => SIZE_MAP[props.size]};
  border: 3px solid transparent;
  border-top-color: ${props => props.hdrEnabled 
    ? THEME_SETTINGS.DARK.hdrColors[props.color]
    : THEME_SETTINGS.DARK[props.color]};
  border-radius: 50%;
  animation: ${props => spinAnimation(props.powerMode)} 
    ${props => ANIMATION_TIMINGS.POWER_MODES[props.powerMode === 'high' 
      ? 'HIGH_PERFORMANCE' 
      : props.powerMode === 'low' 
        ? 'POWER_SAVER' 
        : 'BALANCED'].transitionMultiplier * 1000}ms
    ${ANIMATION_TIMINGS.EASING.DEFAULT} infinite;
  transform-origin: center center;
  backface-visibility: hidden;
  perspective: 1000px;
  will-change: transform;
`;

// HDR-aware text component
const LoadingText = styled.span<{ size: string; hdrEnabled: boolean }>`
  margin-top: ${props => props.size === 'small' ? '8px' : '16px'};
  color: ${props => props.hdrEnabled 
    ? THEME_SETTINGS.DARK.hdrColors.primary 
    : THEME_SETTINGS.DARK.text};
  font-size: ${props => props.size === 'small' ? '14px' : '16px'};
`;

/**
 * Performance-optimized loading component with power and HDR awareness
 */
const Loading: React.FC<LoadingProps> = React.memo(({
  size = 'medium',
  color = 'primary',
  overlay = false,
  text,
  powerMode = 'balanced',
  hdrEnabled = false,
  className
}) => {
  // Normalize power mode for animation timing
  const normalizedPowerMode = powerMode === 'high' 
    ? 'HIGH_PERFORMANCE' 
    : powerMode === 'low' 
      ? 'POWER_SAVER' 
      : 'BALANCED';

  return (
    <LoadingContainer overlay={overlay} className={className}>
      <LoadingSpinner
        size={size}
        color={color}
        powerMode={normalizedPowerMode}
        hdrEnabled={hdrEnabled}
        aria-label="Loading"
        role="progressbar"
      />
      {text && (
        <LoadingText 
          size={size}
          hdrEnabled={hdrEnabled}
          aria-live="polite"
        >
          {text}
        </LoadingText>
      )}
    </LoadingContainer>
  );
});

Loading.displayName = 'Loading';

export default Loading;