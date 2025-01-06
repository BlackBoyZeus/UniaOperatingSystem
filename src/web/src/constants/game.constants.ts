import { RenderQuality } from '../interfaces/game.interface';

/**
 * Core UI constraints and defaults for game display configuration
 * Ensures consistent rendering across different display configurations
 */
export const GAME_UI = {
    /** Minimum supported screen width in pixels (HD) */
    MIN_SCREEN_WIDTH: 1280 as const,
    
    /** Minimum supported screen height in pixels (HD) */
    MIN_SCREEN_HEIGHT: 720 as const,
    
    /** Standard aspect ratio for game display */
    ASPECT_RATIO: '16:9' as const,
    
    /** Default render quality setting */
    DEFAULT_QUALITY: RenderQuality.HIGH as const
} as const;

/**
 * Performance thresholds and targets for game engine optimization
 * Based on technical specifications requirements
 */
export const GAME_PERFORMANCE = {
    /** Target frame rate for optimal gameplay (≥60 FPS requirement) */
    TARGET_FPS: 60 as const,
    
    /** Minimum acceptable frame rate before quality degradation */
    MIN_FPS: 30 as const,
    
    /** Maximum allowed frame time in milliseconds (1000ms/60fps) */
    MAX_FRAME_TIME: 16.6 as const
} as const;

/**
 * Configuration settings for LiDAR visualization overlay
 * Synchronized with LiDAR processing pipeline specifications
 */
export const LIDAR_OVERLAY = {
    /** LiDAR data update rate in Hz (matches 30Hz scan rate) */
    UPDATE_RATE: 30 as const,
    
    /** Point size for LiDAR visualization in pixels */
    POINT_SIZE: 2 as const,
    
    /** Default state for LiDAR overlay visibility */
    DEFAULT_ENABLED: true as const
} as const;

/**
 * Interface for resolution configuration
 */
interface IResolution {
    readonly width: number;
    readonly height: number;
}

/**
 * Standardized resolution presets for different display modes
 * Ensures consistent rendering across different screen configurations
 */
export const RESOLUTION_PRESETS = {
    /** HD resolution preset (1280x720) */
    HD: Object.freeze({ width: 1280, height: 720 }),
    
    /** Full HD resolution preset (1920x1080) */
    FHD: Object.freeze({ width: 1920, height: 1080 }),
    
    /** Quad HD resolution preset (2560x1440) */
    QHD: Object.freeze({ width: 2560, height: 1440 })
} as const;

// Type assertions for better TypeScript support
export type GameUI = typeof GAME_UI;
export type GamePerformance = typeof GAME_PERFORMANCE;
export type LidarOverlay = typeof LIDAR_OVERLAY;
export type ResolutionPresets = typeof RESOLUTION_PRESETS;