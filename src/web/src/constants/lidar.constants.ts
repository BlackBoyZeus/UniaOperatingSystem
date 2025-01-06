/**
 * Core LiDAR scanning configuration constants for the TALD UNIA platform.
 * Defines parameters for scanning, visualization, and performance optimization.
 * @version 1.0.0
 */

/**
 * Core LiDAR scanning configuration parameters matching technical specifications
 */
export const LIDAR_SCAN_SETTINGS = {
    /** Scan frequency in Hz (30Hz continuous scanning) */
    SCAN_FREQUENCY: 30,
    
    /** Scan resolution in centimeters (0.01cm precision) */
    RESOLUTION: 0.01,
    
    /** Maximum effective scanning range in meters */
    MAX_RANGE: 5.0,
    
    /** Minimum effective scanning range in meters */
    MIN_RANGE: 0.1,
    
    /** Maximum allowed processing latency in milliseconds */
    PROCESSING_LATENCY_LIMIT: 50
} as const;

/**
 * LiDAR visualization parameters for point cloud rendering
 */
export const LIDAR_VISUALIZATION = {
    /** Size of individual points in pixels */
    POINT_SIZE: 2,
    
    /** Color scheme for distance-based point coloring */
    COLOR_SCHEME: {
        /** Color for points in near range (0-1.67m) */
        NEAR: '#FF0000',
        
        /** Color for points in mid range (1.67-3.33m) */
        MID: '#00FF00',
        
        /** Color for points in far range (3.33-5m) */
        FAR: '#0000FF'
    },
    
    /** Default point cloud opacity (0-1) */
    OPACITY: 0.8,
    
    /** Visualization refresh rate in Hz */
    REFRESH_RATE: 60
} as const;

/**
 * Performance optimization settings for LiDAR processing
 */
export const LIDAR_PERFORMANCE = {
    /** Maximum number of points processed per scan */
    MAX_POINTS_PER_SCAN: 1_200_000,
    
    /** Buffer size for point cloud data in bytes (1MB) */
    BUFFER_SIZE: 1_048_576,
    
    /** Update interval in milliseconds (derived from 30Hz scan rate) */
    UPDATE_INTERVAL: 33.33
} as const;

// Type definitions for better TypeScript support
export type LidarScanSettings = typeof LIDAR_SCAN_SETTINGS;
export type LidarVisualization = typeof LIDAR_VISUALIZATION;
export type LidarPerformance = typeof LIDAR_PERFORMANCE;
export type ColorScheme = typeof LIDAR_VISUALIZATION.COLOR_SCHEME;