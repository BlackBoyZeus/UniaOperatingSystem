import React, { useCallback, useMemo, useState } from 'react';
import styled from 'styled-components'; // ^6.0.0
import Icon from './Icon';
import { 
  ANIMATION_TIMINGS, 
  UI_INTERACTIONS, 
  THEME_SETTINGS, 
  PERFORMANCE_THRESHOLDS 
} from '../../constants/ui.constants';

// Types
type PowerMode = keyof typeof ANIMATION_TIMINGS.POWER_MODES;

interface DropdownProps {
  options: Array<{ value: string; label: string }>;
  value: string;
  onChange: (value: string) => void;
  powerMode?: PowerMode;
  hdrEnabled?: boolean;
  touchThreshold?: number;
  disabled?: boolean;
  className?: string;
  width?: number;
}

// Styled Components with GPU acceleration and HDR support
const DropdownContainer = styled.div<{
  width: number;
  disabled: boolean;
  hdrEnabled: boolean;
}>`
  position: relative;
  width: ${props => props.width}px;
  user-select: none;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  opacity: ${props => props.disabled ? 0.5 : 1};
  color-space: ${props => props.hdrEnabled ? 'display-p3' : 'srgb'};
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000;
  will-change: transform, opacity;
  
  /* Touch optimizations */
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
`;

const DropdownTrigger = styled.div<{
  isOpen: boolean;
  powerMode: PowerMode;
  hdrEnabled: boolean;
}>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  background: ${props => props.hdrEnabled 
    ? `color(display-p3 ${props.theme.hdrColors.background})` 
    : props.theme.colors.background};
  border-radius: 8px;
  border: 1px solid ${props => props.theme.colors.primary}20;
  
  /* GPU-accelerated animations */
  will-change: transform, opacity;
  transform: translateZ(0);
  transition: all ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms 
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
    
  &:active {
    transform: scale(0.98);
  }
`;

const OptionsContainer = styled.div<{
  isOpen: boolean;
  powerMode: PowerMode;
  hdrEnabled: boolean;
}>`
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  margin-top: 4px;
  background: ${props => props.hdrEnabled 
    ? `color(display-p3 ${props.theme.hdrColors.background})` 
    : props.theme.colors.background};
  border-radius: 8px;
  border: 1px solid ${props => props.theme.colors.primary}20;
  opacity: ${props => props.isOpen ? 1 : 0};
  transform: translateY(${props => props.isOpen ? '0' : '-10px'});
  pointer-events: ${props => props.isOpen ? 'auto' : 'none'};
  
  /* GPU acceleration and power-aware animations */
  will-change: transform, opacity;
  transform: translateZ(0);
  transition: all ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms 
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
  
  /* Performance monitoring */
  &::after {
    content: '';
    display: none;
    animation-timeline: scroll();
    animation-range: entry 0% cover ${PERFORMANCE_THRESHOLDS.FRAME_BUDGET}ms;
  }
`;

const Option = styled.div<{
  isSelected: boolean;
  hdrEnabled: boolean;
}>`
  padding: 12px;
  background: ${props => props.isSelected 
    ? props.hdrEnabled 
      ? `color(display-p3 ${props.theme.hdrColors.primary}20)` 
      : `${props.theme.colors.primary}20`
    : 'transparent'};
  
  &:hover {
    background: ${props => props.hdrEnabled 
      ? `color(display-p3 ${props.theme.hdrColors.primary}10)` 
      : `${props.theme.colors.primary}10`};
  }
  
  /* Touch optimizations */
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
`;

const Dropdown: React.FC<DropdownProps> = React.memo(({
  options,
  value,
  onChange,
  powerMode = 'BALANCED',
  hdrEnabled = false,
  touchThreshold = UI_INTERACTIONS.TOUCH_THRESHOLDS.TAP,
  disabled = false,
  className,
  width = 200
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [touchStart, setTouchStart] = useState<number | null>(null);

  // Memoized selected option
  const selectedOption = useMemo(() => 
    options.find(opt => opt.value === value) || options[0],
    [options, value]
  );

  // Touch handlers with threshold checking
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (disabled) return;
    setTouchStart(e.touches[0].clientY);
  }, [disabled]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!touchStart) return;
    
    const diff = Math.abs(e.touches[0].clientY - touchStart);
    if (diff > touchThreshold) {
      setTouchStart(null);
    }
  }, [touchStart, touchThreshold]);

  const handleTouchEnd = useCallback(() => {
    if (touchStart && !disabled) {
      setIsOpen(prev => !prev);
    }
    setTouchStart(null);
  }, [touchStart, disabled]);

  // Option selection handler
  const handleOptionSelect = useCallback((optionValue: string) => {
    if (disabled) return;
    onChange(optionValue);
    setIsOpen(false);
  }, [onChange, disabled]);

  // Close dropdown on outside click
  React.useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (!(e.target as Element).closest('.dropdown-container')) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return (
    <DropdownContainer
      className={`dropdown-container ${className || ''}`}
      width={width}
      disabled={disabled}
      hdrEnabled={hdrEnabled}
    >
      <DropdownTrigger
        isOpen={isOpen}
        powerMode={powerMode}
        hdrEnabled={hdrEnabled}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onClick={() => !disabled && setIsOpen(prev => !prev)}
      >
        <span>{selectedOption.label}</span>
        <Icon 
          name="chevron-down"
          size={16}
          animate={isOpen}
          powerMode={powerMode}
        />
      </DropdownTrigger>

      <OptionsContainer
        isOpen={isOpen}
        powerMode={powerMode}
        hdrEnabled={hdrEnabled}
      >
        {options.map(option => (
          <Option
            key={option.value}
            isSelected={option.value === value}
            hdrEnabled={hdrEnabled}
            onClick={() => handleOptionSelect(option.value)}
          >
            {option.label}
          </Option>
        ))}
      </OptionsContainer>
    </DropdownContainer>
  );
});

Dropdown.displayName = 'Dropdown';

export type { DropdownProps };
export default Dropdown;