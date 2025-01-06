import React, { useEffect, useCallback } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { usePowerMode } from '@gaming-ui/power-mode';
import { Button, ButtonProps } from './Button';
import { modal } from '../../styles/components.css';

// Modal component props interface
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  size?: 'small' | 'medium' | 'large';
  closeOnOverlayClick?: boolean;
  enableHaptics?: boolean;
  powerMode?: 'performance' | 'balanced' | 'powersave';
  useHDR?: boolean;
}

// Styled components with HDR and power-aware features
const StyledOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: color-mix(in display-p3, var(--color-background) 80%, transparent);
  backdrop-filter: blur(4px);
  z-index: 999;
  will-change: opacity;
  content-visibility: auto;
  isolation: isolate;
`;

const StyledModal = styled(motion.div)<{ size: string; useHDR: boolean }>`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: var(--animation-gpu) translate(-50%, -50%);
  background: ${props => props.useHDR ? 
    'color(display-p3 var(--color-surface))' : 
    'var(--color-surface)'};
  border-radius: 12px;
  padding: calc(var(--spacing-unit) * 3);
  max-width: 90vw;
  max-height: 90vh;
  overflow-y: auto;
  z-index: 1000;
  box-shadow: ${props => props.useHDR ? 'var(--effect-glow)' : 'none'};
  will-change: transform, opacity;
  backface-visibility: hidden;
  content-visibility: auto;
  color-scheme: ${props => props.useHDR ? 'normal hdr' : 'normal'};

  width: ${props => {
    switch (props.size) {
      case 'small': return '400px';
      case 'large': return '800px';
      default: return '600px';
    }
  }};

  @media (dynamic-range: high) {
    background: color(display-p3 var(--color-surface));
    box-shadow: var(--effect-glow);
  }
`;

const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: calc(var(--spacing-unit) * 2);
`;

const ModalTitle = styled.h2`
  font-family: var(--font-family-gaming);
  font-feature-settings: var(--font-feature-gaming);
  margin: 0;
  color: ${props => props.useHDR ? 
    'color(display-p3 var(--color-primary-hdr))' : 
    'var(--color-primary)'};
`;

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'medium',
  closeOnOverlayClick = true,
  enableHaptics = true,
  powerMode = 'balanced',
  useHDR = true,
}) => {
  // Get power-aware animation configuration
  const getAnimationConfig = useCallback(() => {
    switch (powerMode) {
      case 'performance':
        return {
          duration: 0.15,
          ease: 'easeOut',
          type: 'tween',
        };
      case 'powersave':
        return {
          duration: 0.3,
          ease: 'linear',
          type: 'tween',
        };
      default:
        return {
          duration: 0.2,
          ease: 'easeInOut',
          type: 'spring',
          stiffness: 300,
          damping: 30,
        };
    }
  }, [powerMode]);

  // Handle overlay click with haptic feedback
  const handleOverlayClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (closeOnOverlayClick) {
      if (enableHaptics && window.navigator?.vibrate) {
        window.navigator.vibrate(50);
      }
      onClose();
    }
  }, [closeOnOverlayClick, enableHaptics, onClose]);

  // Handle escape key press
  useEffect(() => {
    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (enableHaptics && window.navigator?.vibrate) {
          window.navigator.vibrate(50);
        }
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscapeKey);
    }

    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isOpen, onClose, enableHaptics]);

  const animationConfig = getAnimationConfig();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <StyledOverlay
            className={modal}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={animationConfig}
            onClick={handleOverlayClick}
          />
          <StyledModal
            size={size}
            useHDR={useHDR}
            initial={{ opacity: 0, scale: 0.95, y: '-45%' }}
            animate={{ opacity: 1, scale: 1, y: '-50%' }}
            exit={{ opacity: 0, scale: 0.95, y: '-45%' }}
            transition={animationConfig}
          >
            <ModalHeader>
              <ModalTitle>{title}</ModalTitle>
              <Button
                variant="secondary"
                size="small"
                onClick={onClose}
                enableHaptic={enableHaptics}
                hdrMode={useHDR ? 'enabled' : 'disabled'}
                powerSaveAware={powerMode === 'powersave'}
              >
                ✕
              </Button>
            </ModalHeader>
            {children}
          </StyledModal>
        </>
      )}
    </AnimatePresence>
  );
};

export type { ModalProps };