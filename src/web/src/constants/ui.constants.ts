/**
 * @file UI Constants and Configuration
 * @version 1.0.0
 * @description Core UI constants for TALD UNIA web interface with HDR, GPU acceleration,
 * and power-aware optimizations
 */

import { UserPreferencesType } from '../interfaces/user.interface';

/**
 * Theme configuration type with HDR color support
 */
interface ThemeConfig {
  background: string;
  primary: string;
  secondary: string;
  text: string;
  accent: string;
  hdrColors: {
    primary: string;
    secondary: string;
    accent: string;
  };
  contrast: {
    high: string;
    medium: string;
    low: string;
  };
}

/**
 * Core theme settings with HDR color space support
 * Implements P3 color space for enhanced visual fidelity
 */
export const THEME_SETTINGS: Record<string, ThemeConfig> = {
  DARK: {
    background: '#121212',
    primary: '#00FF00',
    secondary: '#0000FF',
    text: '#FFFFFF',
    accent: '#FF0000',
    hdrColors: {
      primary: 'color(display-p3 0 1 0)',
      secondary: 'color(display-p3 0 0 1)',
      accent: 'color(display-p3 1 0 0)'
    },
    contrast: {
      high: 'rgba(255, 255, 255, 0.87)',
      medium: 'rgba(255, 255, 255, 0.60)',
      low: 'rgba(255, 255, 255, 0.38)'
    }
  },
  LIGHT: {
    background: '#FFFFFF',
    primary: '#00CC00',
    secondary: '#0000CC',
    text: '#000000',
    accent: '#CC0000',
    hdrColors: {
      primary: 'color(display-p3 0 0.8 0)',
      secondary: 'color(display-p3 0 0 0.8)',
      accent: 'color(display-p3 0.8 0 0)'
    },
    contrast: {
      high: 'rgba(0, 0, 0, 0.87)',
      medium: 'rgba(0, 0, 0, 0.60)',
      low: 'rgba(0, 0, 0, 0.38)'
    }
  }
};

/**
 * Layout constants based on 8px grid system
 * Implements safe zones for various device aspects
 */
export const LAYOUT_CONSTANTS = {
  GRID_BASE: 8,
  SAFE_ZONE: {
    top: 24,
    bottom: 24,
    left: 16,
    right: 16
  },
  SUBGRID: {
    spacing: 4,
    columns: 12,
    gutters: 16
  },
  ASPECT_RATIOS: {
    WIDE: '21:9',
    STANDARD: '16:9',
    PORTABLE: '16:10'
  },
  BREAKPOINTS: {
    xs: 320,
    sm: 600,
    md: 960,
    lg: 1280,
    xl: 1920
  }
};

/**
 * GPU-optimized animation timings and power-aware adjustments
 * Ensures consistent 60 FPS with power state management
 */
export const ANIMATION_TIMINGS = {
  TRANSITION_DURATION: 150,
  ANIMATION_FPS: 60,
  MIN_FRAME_TIME: 16.67, // 1000ms / 60fps
  EASING: {
    DEFAULT: 'cubic-bezier(0.4, 0, 0.2, 1)',
    ACCELERATED: 'cubic-bezier(0.4, 0, 1, 1)',
    DECELERATED: 'cubic-bezier(0, 0, 0.2, 1)',
    SHARP: 'cubic-bezier(0.4, 0, 0.6, 1)'
  },
  POWER_MODES: {
    HIGH_PERFORMANCE: {
      fps: 60,
      transitionMultiplier: 1.0
    },
    BALANCED: {
      fps: 45,
      transitionMultiplier: 1.2
    },
    POWER_SAVER: {
      fps: 30,
      transitionMultiplier: 1.5
    }
  }
};

/**
 * UI interaction constants optimized for gaming input
 * Implements sub-16ms input latency requirements
 */
export const UI_INTERACTIONS = {
  INPUT_LATENCY_LIMIT: 16,
  TOUCH_THRESHOLDS: {
    TAP: 10,
    DRAG: 20,
    SWIPE: 50,
    PRECISION: 5
  },
  GESTURE_TIMINGS: {
    DOUBLE_TAP: 300,
    LONG_PRESS: 500,
    SWIPE_DURATION: 200,
    MOMENTUM_DURATION: 1000
  },
  HAPTIC_FEEDBACK: {
    LIGHT: 10,
    MEDIUM: 15,
    HEAVY: 25
  }
};

/**
 * Typography system with gaming-optimized font settings
 * Supports variable font weights and bitmap optimization
 */
export const TYPOGRAPHY = {
  FONT_FAMILY: 'gaming-font, sans-serif',
  BASE_SIZE: 16,
  SCALE_RATIO: 1.25,
  LINE_HEIGHT: 1.5,
  VARIABLE_WEIGHTS: {
    LIGHT: 300,
    REGULAR: 400,
    MEDIUM: 500,
    BOLD: 700,
    BLACK: 900
  },
  BITMAP_OPTIMIZED_SIZES: [12, 14, 16, 18, 24, 32, 48]
};

/**
 * Performance monitoring thresholds
 * Ensures consistent frame timing and power efficiency
 */
export const PERFORMANCE_THRESHOLDS = {
  FRAME_BUDGET: 16.67,
  ANIMATION_BUDGET: 8,
  INTERACTION_BUDGET: 8,
  RENDER_BUDGET: 12,
  POWER_USAGE: {
    HIGH: 100,
    MEDIUM: 75,
    LOW: 50
  }
};

/**
 * Z-index stacking order management
 */
export const Z_INDEX = {
  BACKGROUND: -1,
  BASE: 0,
  OVERLAY: 100,
  MODAL: 200,
  TOOLTIP: 300,
  NOTIFICATION: 400,
  CRITICAL: 500
};