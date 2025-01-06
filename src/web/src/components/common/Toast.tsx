import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import Icon from './Icon';
import { ANIMATION_TIMINGS, THEME_SETTINGS, Z_INDEX } from '../../constants/ui.constants';

/**
 * Toast type definition for different notification states
 */
type ToastType = 'success' | 'error' | 'warning' | 'info';

/**
 * Toast position options
 */
type ToastPosition = 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';

/**
 * Power mode options for animation optimization
 */
type PowerMode = keyof typeof ANIMATION_TIMINGS.POWER_MODES;

/**
 * Priority levels for toast notifications
 */
type Priority = 'high' | 'normal';

/**
 * Props interface for the Toast component
 */
interface ToastProps {
  message: string;
  type?: ToastType;
  duration?: number;
  onClose?: () => void;
  position?: ToastPosition;
  powerMode?: PowerMode;
  hdrEnabled?: boolean;
  priority?: Priority;
  fleetSync?: boolean;
}

/**
 * Icon mapping for different toast types with HDR colors
 */
const TOAST_ICONS: Record<ToastType, string> = {
  success: 'M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z',
  error: 'M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z',
  warning: 'M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z',
  info: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z'
};

/**
 * Styled toast container with GPU acceleration and HDR support
 */
const StyledToast = styled(motion.div)<{
  type: ToastType;
  position: ToastPosition;
  hdrEnabled: boolean;
  priority: Priority;
}>`
  position: fixed;
  min-width: 300px;
  max-width: 400px;
  padding: ${props => props.theme.GRID_BASE * 2}px;
  border-radius: 8px;
  background: ${props => THEME_SETTINGS[props.theme].background};
  color: ${props => THEME_SETTINGS[props.theme].text};
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  gap: ${props => props.theme.GRID_BASE}px;
  z-index: ${props => props.priority === 'high' ? Z_INDEX.CRITICAL : Z_INDEX.NOTIFICATION};
  
  /* GPU acceleration optimizations */
  will-change: transform, opacity;
  pointer-events: auto;
  transform-style: preserve-3d;
  backface-visibility: hidden;
  
  /* HDR color enhancements */
  filter: ${props => props.hdrEnabled ? 'contrast(1.1)' : 'none'};
  transition: filter 0.2s ease-out;
  
  /* Position styling */
  ${props => {
    const positions = {
      'top-right': 'top: 24px; right: 24px;',
      'top-left': 'top: 24px; left: 24px;',
      'bottom-right': 'bottom: 24px; right: 24px;',
      'bottom-left': 'bottom: 24px; left: 24px;'
    };
    return positions[props.position];
  }}
`;

/**
 * Power-aware animation variants
 */
const toastVariants = {
  initial: {
    opacity: 0,
    y: -20,
    scale: 0.95
  },
  animate: (powerMode: PowerMode) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: ANIMATION_TIMINGS.POWER_MODES[powerMode].transitionMultiplier * 0.2,
      ease: ANIMATION_TIMINGS.EASING.DEFAULT
    }
  }),
  exit: (powerMode: PowerMode) => ({
    opacity: 0,
    scale: 0.95,
    transition: {
      duration: ANIMATION_TIMINGS.POWER_MODES[powerMode].transitionMultiplier * 0.15,
      ease: ANIMATION_TIMINGS.EASING.ACCELERATED
    }
  })
};

/**
 * High-performance Toast component with HDR and power-aware features
 */
const Toast: React.FC<ToastProps> = React.memo(({
  message,
  type = 'info',
  duration = 3000,
  onClose,
  position = 'top-right',
  powerMode = 'BALANCED',
  hdrEnabled = true,
  priority = 'normal',
  fleetSync = false
}) => {
  // Auto-dismiss timer with power mode consideration
  React.useEffect(() => {
    if (duration) {
      const adjustedDuration = duration * ANIMATION_TIMINGS.POWER_MODES[powerMode].transitionMultiplier;
      const timer = setTimeout(() => {
        onClose?.();
      }, adjustedDuration);
      
      return () => clearTimeout(timer);
    }
  }, [duration, onClose, powerMode]);

  // Fleet synchronization effect
  React.useEffect(() => {
    if (fleetSync) {
      // Fleet-wide notification sync would be implemented here
      // This is a placeholder for the actual implementation
      const syncToFleet = async () => {
        try {
          // Sync notification to fleet members
        } catch (error) {
          console.error('Fleet sync failed:', error);
        }
      };

      syncToFleet();
    }
  }, [fleetSync]);

  return (
    <AnimatePresence mode="wait">
      <StyledToast
        type={type}
        position={position}
        hdrEnabled={hdrEnabled}
        priority={priority}
        variants={toastVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        custom={powerMode}
        layout
      >
        <Icon
          name={TOAST_ICONS[type]}
          size={24}
          color={THEME_SETTINGS[type].hdrColors.primary}
          powerMode={powerMode}
        />
        <span>{message}</span>
      </StyledToast>
    </AnimatePresence>
  );
});

Toast.displayName = 'Toast';

export type { ToastProps, ToastType, ToastPosition, Priority };
export default Toast;