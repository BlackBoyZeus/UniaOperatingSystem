import React, { useCallback, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import debounce from 'lodash/debounce';
import { button } from '../../styles/components.css';

// Interface for Button component props
interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'accent';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  enableHaptic?: boolean;
  hdrMode?: 'auto' | 'enabled' | 'disabled';
  powerSaveAware?: boolean;
}

// Styled button component with HDR and power-aware features
const StyledButton = styled.button<ButtonProps>`
  ${button}
  font-family: var(--font-family-gaming);
  font-feature-settings: var(--font-feature-gaming);
  backface-visibility: hidden;
  transform: var(--animation-gpu);
  will-change: ${props => props.powerSaveAware ? 'auto' : 'transform, background-color'};
  transition: transform ${props => props.powerSaveAware ? 'var(--animation-duration-power-save)' : 'var(--animation-duration-normal)'} cubic-bezier(0.4, 0, 0.2, 1),
              background-color 100ms linear;
  color-scheme: dark light;
  user-select: none;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
  content-visibility: auto;
  contain: content;

  /* Variant styles with HDR color support */
  ${props => {
    const colorVar = props.hdrMode === 'enabled' ? '-hdr' : '';
    switch (props.variant) {
      case 'secondary':
        return `
          background: color(display-p3 var(--color-secondary${colorVar}));
          color: color(display-p3 1 1 1);
        `;
      case 'accent':
        return `
          background: color(display-p3 var(--color-accent${colorVar}));
          color: color(display-p3 1 1 1);
        `;
      default:
        return `
          background: color(display-p3 var(--color-primary${colorVar}));
          color: color(display-p3 1 1 1);
        `;
    }
  }}

  /* Size variants */
  ${props => {
    switch (props.size) {
      case 'small':
        return `
          padding: calc(var(--spacing-unit)) calc(var(--spacing-unit) * 2);
          font-size: 0.875rem;
        `;
      case 'large':
        return `
          padding: calc(var(--spacing-unit) * 2) calc(var(--spacing-unit) * 4);
          font-size: 1.125rem;
        `;
      default:
        return `
          padding: calc(var(--spacing-unit) * 1.5) calc(var(--spacing-unit) * 3);
          font-size: 1rem;
        `;
    }
  }}

  /* Full width option */
  width: ${props => props.fullWidth ? '100%' : 'auto'};

  /* Interactive states */
  &:hover:not(:disabled) {
    transform: ${props => props.powerSaveAware ? 'none' : 'var(--animation-gpu) scale(0.98)'};
    filter: brightness(1.1);
    box-shadow: ${props => props.hdrMode !== 'disabled' ? 'var(--effect-glow)' : 'none'};
  }

  &:active:not(:disabled) {
    transform: ${props => props.powerSaveAware ? 'none' : 'var(--animation-gpu) scale(0.95)'};
    filter: brightness(0.9);
  }

  /* Disabled state */
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    filter: none;
    box-shadow: none;
  }

  /* HDR display support */
  @media (dynamic-range: high) {
    ${props => props.hdrMode === 'auto' && `
      box-shadow: var(--effect-glow);
      background: color(display-p3 var(--color-${props.variant || 'primary'}-hdr));
    `}
  }

  /* Power save mode optimizations */
  @media (prefers-reduced-motion: reduce) {
    ${props => props.powerSaveAware && `
      transition: none;
      transform: none;
      will-change: auto;
    `}
  }
`;

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  fullWidth = false,
  onClick,
  type = 'button',
  className,
  enableHaptic = true,
  hdrMode = 'auto',
  powerSaveAware = true,
  ...props
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  
  // Debounced click handler with power-aware optimizations
  const handleClick = useCallback(
    debounce((event: React.MouseEvent<HTMLButtonElement>) => {
      if (disabled) return;
      
      if (type === 'submit') {
        event.preventDefault();
      }

      // Trigger haptic feedback if enabled
      if (enableHaptic && window.navigator?.vibrate) {
        window.navigator.vibrate(50);
      }

      onClick?.(event);
    }, 50),
    [disabled, onClick, type, enableHaptic]
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      handleClick.cancel();
    };
  }, [handleClick]);

  return (
    <StyledButton
      ref={buttonRef}
      type={type}
      disabled={disabled}
      variant={variant}
      size={size}
      fullWidth={fullWidth}
      className={className}
      onClick={handleClick}
      hdrMode={hdrMode}
      powerSaveAware={powerSaveAware}
      {...props}
    >
      {children}
    </StyledButton>
  );
};

export type { ButtonProps };