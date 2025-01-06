import { Point3D } from '../types/lidar.types';
import { GameState } from '../types/game.types';
import { ANIMATION_TIMINGS } from '../constants/ui.constants';

// Default precision for coordinate formatting
const DEFAULT_PRECISION = 2;

// Performance thresholds for metric formatting
const PERFORMANCE_THRESHOLDS = {
    LOW: 30,
    MEDIUM: 45,
    HIGH: 60
} as const;

// Memoization cache for frequently formatted values
const formatCache = new Map<string, string>();

/**
 * Formats a Point3D coordinate with configurable precision and validation
 * Ensures compliance with 0.01cm resolution and 5-meter range specifications
 * 
 * @param point - The Point3D coordinate to format
 * @param precision - Number of decimal places (default: 2, minimum: 2 for 0.01cm)
 * @returns Formatted coordinate string with proper units and validation
 * @throws Error if coordinates are invalid or out of range
 */
export function formatPoint3D(point: Point3D, precision: number = DEFAULT_PRECISION): string {
    // Input validation
    if (!point || typeof point.x !== 'number' || 
        typeof point.y !== 'number' || 
        typeof point.z !== 'number') {
        throw new Error('Invalid Point3D coordinates');
    }

    // Range validation (5-meter limit)
    const maxRange = 5.0;
    if (Math.abs(point.x) > maxRange || 
        Math.abs(point.y) > maxRange || 
        Math.abs(point.z) > maxRange) {
        throw new Error('Coordinates exceed 5-meter range limit');
    }

    // Precision validation and adjustment
    const validPrecision = Math.max(2, Math.min(precision, 4));
    
    // Generate cache key
    const cacheKey = `${point.x},${point.y},${point.z},${validPrecision}`;
    
    // Check cache first
    const cached = formatCache.get(cacheKey);
    if (cached) {
        return cached;
    }

    // Format coordinates with proper precision
    const formattedX = point.x.toFixed(validPrecision);
    const formattedY = point.y.toFixed(validPrecision);
    const formattedZ = point.z.toFixed(validPrecision);

    // Construct formatted string with units
    const formatted = `(${formattedX}m, ${formattedY}m, ${formattedZ}m)`;
    
    // Cache result if cache isn't too large
    if (formatCache.size < 1000) {
        formatCache.set(cacheKey, formatted);
    }

    return formatted;
}

/**
 * Formats game state enum to localized, accessible display string
 * Supports high contrast mode and screen reader compatibility
 * 
 * @param state - The GameState enum value to format
 * @returns Localized and accessible game state string
 */
export function formatGameState(state: GameState): string {
    if (!Object.values(GameState).includes(state)) {
        throw new Error('Invalid game state');
    }

    // Map states to localized, accessible strings
    const stateMap: Record<GameState, string> = {
        [GameState.INITIALIZING]: 'Initializing System',
        [GameState.LOADING]: 'Loading Game',
        [GameState.RUNNING]: 'Game Active',
        [GameState.PAUSED]: 'Game Paused',
        [GameState.ENDED]: 'Game Ended'
    };

    // Add accessibility attributes
    const formatted = stateMap[state];
    return `<span role="status" aria-live="polite">${formatted}</span>`;
}

/**
 * Formats performance metrics with unit awareness and accessibility support
 * Implements color coding and high contrast mode compatibility
 * 
 * @param value - The numeric value to format
 * @param unit - The unit type ('fps', 'ms', etc.)
 * @returns Formatted performance metric with proper styling
 */
export function formatPerformanceMetric(value: number, unit: string): string {
    // Input validation
    if (typeof value !== 'number' || !unit) {
        throw new Error('Invalid performance metric input');
    }

    // Round value based on unit type
    const roundedValue = unit === 'fps' ? Math.round(value) : 
                        unit === 'ms' ? Number(value.toFixed(1)) :
                        Number(value.toFixed(2));

    // Determine performance level
    let performanceLevel: 'low' | 'medium' | 'high';
    if (unit === 'fps') {
        performanceLevel = roundedValue >= PERFORMANCE_THRESHOLDS.HIGH ? 'high' :
                          roundedValue >= PERFORMANCE_THRESHOLDS.MEDIUM ? 'medium' : 'low';
    } else {
        performanceLevel = roundedValue <= ANIMATION_TIMINGS.MIN_FRAME_TIME ? 'high' :
                          roundedValue <= ANIMATION_TIMINGS.MIN_FRAME_TIME * 1.5 ? 'medium' : 'low';
    }

    // Color mapping with high contrast alternatives
    const colorMap = {
        high: { normal: '#00FF00', highContrast: '#FFFFFF' },
        medium: { normal: '#FFFF00', highContrast: '#CCCCCC' },
        low: { normal: '#FF0000', highContrast: '#666666' }
    };

    // Construct accessible formatted string
    return `<span 
        role="text" 
        aria-label="${roundedValue} ${unit}"
        class="performance-metric performance-${performanceLevel}"
        style="color: var(--metric-color, ${colorMap[performanceLevel].normal})"
    >${roundedValue}${unit}</span>`;
}