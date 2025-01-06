import React, { useCallback, useRef, useEffect } from 'react';
import styled from '@emotion/styled';
import { Slider } from '@mui/material';
import { colors, HDR_COLORS } from '../../config/theme.config';
import { ANIMATION_TIMINGS, UI_INTERACTIONS, PERFORMANCE_THRESHOLDS } from '../../constants/ui.constants';

/**
 * Power mode type for animation optimization
 */
type PowerMode = keyof typeof ANIMATION_TIMINGS.POWER_MODES;

/**
 * Enhanced props interface for the GPU-accelerated slider component
 */
interface SliderProps {
  value: number;
  onChange: (value: number, latency: number) => void;
  min: number;
  max: number;
  step: number;
  label: string;
  disabled?: boolean;
  showValue?: boolean;
  powerMode?: PowerMode;
  criticalSetting?: boolean;
}

/**
 * GPU-accelerated container with HDR color support
 */
const StyledSliderContainer = styled.div`
  position: relative;
  width: 100%;
  padding: ${ANIMATION_TIMINGS.MIN_FRAME_TIME}px 0;
  transform: translate3d(0, 0, 0);
  will-change: transform;
  color-space: display-p3;
  -webkit-font-smoothing: antialiased;
`;

/**
 * Enhanced Material-UI slider with power-aware animations
 */
const StyledSlider = styled(Slider)<{ powerMode?: PowerMode }>`
  color: ${props => props.theme.colorDepth.hdrEnabled ? HDR_COLORS.primary.main : colors.primary.main};
  transition-property: all;
  transition-timing-function: ${ANIMATION_TIMINGS.EASING.DEFAULT};
  transition-duration: ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode || 'BALANCED'].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms;
  
  & .MuiSlider-thumb {
    transform: translate3d(0, 0, 0);
    will-change: transform;
    
    &:hover {
      box-shadow: 0 0 0 8px ${props => 
        props.theme.colorDepth.hdrEnabled ? 
        'color(display-p3 0.486 0.302 1 / 0.16)' : 
        'rgba(123, 77, 255, 0.16)'};
    }
  }

  & .MuiSlider-track {
    transition: width ${props => 
      ANIMATION_TIMINGS.POWER_MODES[props.powerMode || 'BALANCED'].transitionMultiplier * 
      ANIMATION_TIMINGS.TRANSITION_DURATION}ms;
  }

  &.Mui-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

/**
 * Value display with HDR color support
 */
const ValueDisplay = styled.div`
  position: absolute;
  right: 0;
  top: 0;
  font-family: ${props => props.theme.typography.gaming.family};
  font-weight: ${props => props.theme.typography.gaming.weights.regular};
  color: ${props => props.theme.colorDepth.hdrEnabled ? 
    'color(display-p3 1 1 1)' : 
    props.theme.palette.text.primary};
  transform: translate3d(0, 0, 0);
`;

/**
 * High-performance slider component with GPU acceleration and power optimization
 */
const CustomSlider: React.FC<SliderProps> = React.memo(({
  value,
  onChange,
  min,
  max,
  step,
  label,
  disabled = false,
  showValue = true,
  powerMode = 'BALANCED',
  criticalSetting = false
}) => {
  const lastUpdateTime = useRef<number>(performance.now());
  const frameRequest = useRef<number>();

  /**
   * Optimized change handler with latency tracking
   */
  const handleChange = useCallback((event: Event, newValue: number) => {
    if (frameRequest.current) {
      cancelAnimationFrame(frameRequest.current);
    }

    frameRequest.current = requestAnimationFrame(() => {
      const currentTime = performance.now();
      const inputLatency = currentTime - lastUpdateTime.current;

      // Validate critical settings
      if (criticalSetting) {
        const validatedValue = Math.max(min, Math.min(max, newValue));
        if (validatedValue !== newValue) {
          return;
        }
      }

      // Log performance warning if latency exceeds threshold
      if (inputLatency > UI_INTERACTIONS.INPUT_LATENCY_LIMIT) {
        console.warn(`Slider input latency exceeded threshold: ${inputLatency.toFixed(2)}ms`);
      }

      onChange(newValue, inputLatency);
      lastUpdateTime.current = currentTime;
    });
  }, [onChange, min, max, criticalSetting]);

  // Cleanup frame requests
  useEffect(() => {
    return () => {
      if (frameRequest.current) {
        cancelAnimationFrame(frameRequest.current);
      }
    };
  }, []);

  return (
    <StyledSliderContainer>
      <StyledSlider
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        powerMode={powerMode}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        role="slider"
        tabIndex={disabled ? -1 : 0}
      />
      {showValue && (
        <ValueDisplay>
          {value}
        </ValueDisplay>
      )}
    </StyledSliderContainer>
  );
});

CustomSlider.displayName = 'CustomSlider';

export type { SliderProps };
export default CustomSlider;