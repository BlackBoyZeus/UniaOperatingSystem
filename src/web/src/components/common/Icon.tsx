import React from 'react';
import styled from 'styled-components';
import { ANIMATION_TIMINGS, PERFORMANCE_THRESHOLDS, UI_INTERACTIONS } from '../../constants/ui.constants';

/**
 * Power mode type for animation optimization
 */
type PowerMode = keyof typeof ANIMATION_TIMINGS.POWER_MODES;

/**
 * Props interface for the Icon component with HDR and power-aware features
 */
interface IconProps {
  name: string;
  size?: number;
  color?: string;
  className?: string;
  animate?: boolean;
  powerMode?: PowerMode;
}

/**
 * Calculate animation duration based on power mode
 */
const getAnimationDuration = (powerMode: PowerMode = 'BALANCED'): number => {
  const baseTransition = ANIMATION_TIMINGS.TRANSITION_DURATION;
  const multiplier = ANIMATION_TIMINGS.POWER_MODES[powerMode].transitionMultiplier;
  return baseTransition * multiplier;
};

/**
 * Generate power-aware animation styles
 */
const getPowerAwareAnimation = (powerMode: PowerMode = 'BALANCED') => {
  const fps = ANIMATION_TIMINGS.POWER_MODES[powerMode].fps;
  const stepDuration = 1000 / fps;
  
  return `
    @keyframes iconPulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.1); }
      100% { transform: scale(1); }
    }
    animation: iconPulse ${stepDuration * 8}ms ${ANIMATION_TIMINGS.EASING.DEFAULT} infinite;
    animation-play-state: var(--animation-play-state, running);
  `;
};

/**
 * Styled SVG component with GPU acceleration and HDR support
 */
const StyledIcon = styled.svg<{
  color?: string;
  animate?: boolean;
  powerMode: PowerMode;
}>`
  display: inline-block;
  vertical-align: middle;
  fill: ${props => props.color || 'currentColor'};
  color-gamut: p3;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000;
  will-change: transform;
  
  /* Power-aware transitions */
  transition: transform ${props => getAnimationDuration(props.powerMode)}ms 
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
  
  /* HDR color space support */
  @supports (color: color(display-p3 0 0 0)) {
    fill: ${props => props.color?.startsWith('color(display-p3') 
      ? props.color 
      : `color(display-p3 ${props.color || 'currentColor'})`};
  }
  
  /* Animation styles */
  ${props => props.animate && getPowerAwareAnimation(props.powerMode)}
  
  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
    animation: none;
  }
  
  /* Performance monitoring */
  &::after {
    content: '';
    display: none;
    animation-timeline: scroll();
    animation-range: entry 0% cover ${PERFORMANCE_THRESHOLDS.FRAME_BUDGET}ms;
  }
`;

/**
 * High-performance Icon component with GPU acceleration and power awareness
 */
const Icon: React.FC<IconProps> = React.memo(({
  name,
  size = 24,
  color,
  className,
  animate = false,
  powerMode = 'BALANCED'
}) => {
  // Performance optimization for re-renders
  const memoizedProps = React.useMemo(() => ({
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    className,
    color,
    animate,
    powerMode,
    'aria-hidden': 'true',
    style: {
      // Ensure sub-16ms input latency
      touchAction: 'manipulation',
      pointerEvents: 'none',
      userSelect: 'none',
    } as React.CSSProperties
  }), [size, className, color, animate, powerMode]);

  return (
    <StyledIcon {...memoizedProps}>
      <path d={name} />
    </StyledIcon>
  );
});

Icon.displayName = 'Icon';

export type { IconProps };
export default Icon;