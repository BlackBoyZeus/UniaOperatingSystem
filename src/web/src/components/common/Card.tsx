import React, { useCallback, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { usePerformanceMonitor } from '@performance-monitor/react';
import { card, powerAwareTransition, hdrColorScheme } from '../../styles/components.css';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 * @external @performance-monitor/react v1.0.0
 */

interface CardProps {
  children: React.ReactNode;
  variant?: 'elevated' | 'outlined';
  onClick?: (event: React.MouseEvent<HTMLDivElement>) => void;
  className?: string;
  interactive?: boolean;
  hdrMode?: 'auto' | 'enabled' | 'disabled';
  powerSaveMode?: boolean;
}

const StyledCard = styled.div<Omit<CardProps, 'children'>>`
  /* Base styles with GPU acceleration */
  background: color(display-p3 var(--color-surface));
  border-radius: calc(var(--spacing-unit) * 1);
  padding: calc(var(--spacing-unit) * 2);
  transform: var(--animation-gpu);
  will-change: transform, opacity;
  backface-visibility: hidden;
  color-scheme: dark light;
  contain: layout style paint;
  touch-action: manipulation;
  -webkit-touch-callout: none;

  /* HDR-aware transitions */
  transition: all var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1);

  /* Variant styles */
  ${({ variant }) =>
    variant === 'elevated' &&
    `
    box-shadow: 0 4px 6px color(display-p3 0 0 0 / 0.1);
    @media (dynamic-range: high) {
      box-shadow: 0 4px 12px color(display-p3 0 0 0 / 0.2);
    }
  `}

  ${({ variant }) =>
    variant === 'outlined' &&
    `
    border: 1px solid color(display-p3 1 1 1 / 0.1);
    @media (dynamic-range: high) {
      border-color: color(display-p3 1 1 1 / 0.15);
    }
  `}

  /* Interactive states */
  ${({ interactive }) =>
    interactive &&
    `
    cursor: pointer;
    &:hover {
      transform: var(--animation-gpu) scale3d(1.02, 1.02, 1);
      box-shadow: 0 8px 12px color(display-p3 0 0 0 / 0.15);
    }
    &:active {
      transform: var(--animation-gpu) scale3d(0.98, 0.98, 1);
    }
  `}

  /* Power-save mode optimizations */
  ${({ powerSaveMode }) =>
    powerSaveMode &&
    `
    transition-duration: var(--animation-duration-power-save);
    will-change: auto;
    &:hover {
      transform: none;
    }
  `}
`;

const useHDRDetection = (mode: CardProps['hdrMode'] = 'auto'): boolean => {
  const [isHDRSupported, setIsHDRSupported] = React.useState(false);

  useEffect(() => {
    if (mode === 'enabled') {
      setIsHDRSupported(true);
      return;
    }
    if (mode === 'disabled') {
      setIsHDRSupported(false);
      return;
    }

    // Auto-detect HDR support
    const mediaQuery = window.matchMedia('(dynamic-range: high)');
    setIsHDRSupported(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setIsHDRSupported(e.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [mode]);

  return isHDRSupported;
};

export const Card: React.FC<CardProps> = ({
  children,
  variant = 'elevated',
  onClick,
  className,
  interactive = false,
  hdrMode = 'auto',
  powerSaveMode = false,
  ...props
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const { trackInteraction } = usePerformanceMonitor();
  const isHDRSupported = useHDRDetection(hdrMode);

  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!interactive || !onClick) return;

      const interactionId = trackInteraction('card_click');
      
      // Apply power-aware animations
      if (cardRef.current && !powerSaveMode) {
        cardRef.current.style.transform = 'var(--animation-gpu) scale3d(0.98, 0.98, 1)';
        setTimeout(() => {
          if (cardRef.current) {
            cardRef.current.style.transform = 'var(--animation-gpu)';
          }
        }, powerSaveMode ? 200 : 100);
      }

      onClick(event);
      trackInteraction(interactionId, 'complete');
    },
    [interactive, onClick, powerSaveMode, trackInteraction]
  );

  return (
    <StyledCard
      ref={cardRef}
      variant={variant}
      onClick={handleClick}
      className={`${card} ${className || ''} ${
        isHDRSupported ? hdrColorScheme : ''
      } ${powerSaveMode ? powerAwareTransition : ''}`}
      interactive={interactive}
      powerSaveMode={powerSaveMode}
      data-hdr={isHDRSupported ? 'enabled' : 'disabled'}
      {...props}
    >
      {children}
    </StyledCard>
  );
};

// Export types for external usage
export type { CardProps };