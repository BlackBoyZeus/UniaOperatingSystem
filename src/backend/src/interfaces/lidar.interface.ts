import { Buffer } from 'buffer'; // version: latest

/**
 * Enum defining scan quality levels for different processing modes and hardware capabilities
 */
export enum ScanQuality {
    HIGH = 'HIGH',       // Maximum quality for detailed environment mapping
    MEDIUM = 'MEDIUM',   // Balanced quality for general use
    LOW = 'LOW'         // Reduced quality for power saving
}

/**
 * Enum defining LiDAR processing modes with different performance and power consumption tradeoffs
 */
export enum ProcessingMode {
    REAL_TIME = 'REAL_TIME',         // Optimized for 30Hz scanning with ≤50ms latency
    HIGH_QUALITY = 'HIGH_QUALITY',    // Maximum accuracy with increased latency
    POWER_SAVE = 'POWER_SAVE'        // Reduced power consumption mode
}

/**
 * Power management modes for LiDAR operation
 */
export enum PowerMode {
    PERFORMANCE = 'PERFORMANCE',  // Maximum performance mode
    BALANCED = 'BALANCED',        // Balanced power and performance
    EFFICIENCY = 'EFFICIENCY'     // Maximum power efficiency
}

/**
 * Calibration data structure for LiDAR sensor
 */
export interface CalibrationData {
    offsetX: number;
    offsetY: number;
    offsetZ: number;
    rotationMatrix: number[][];
    distortionParams: number[];
    timestamp: number;
}

/**
 * Configuration interface for LiDAR hardware and processing settings
 */
export interface ILidarConfig {
    scanRate: number;           // Scan frequency in Hz (max 30Hz)
    resolution: number;         // Spatial resolution in cm (min 0.01cm)
    range: number;             // Maximum scanning range in meters (max 5.0m)
    processingMode: ProcessingMode;
    powerMode: PowerMode;
    calibrationData: CalibrationData;
}

/**
 * Interface for processed point cloud data structure
 */
export interface IPointCloud {
    points: Buffer;            // Binary buffer containing point cloud data
    timestamp: number;         // Scan timestamp in milliseconds
    quality: ScanQuality;      // Scan quality level
    density: number;           // Points per cubic meter
    confidence: number;        // Confidence score [0-1]
}

/**
 * Interface for comprehensive scan operation metadata
 */
export interface IScanMetadata {
    scanId: string;           // Unique identifier for scan operation
    timestamp: number;        // Operation timestamp
    processingTime: number;   // Processing duration in milliseconds
    quality: ScanQuality;     // Achieved scan quality
    errorRate: number;        // Error rate [0-1]
    powerConsumption: number; // Power consumption in watts
}

/**
 * Interface defining comprehensive LiDAR processing pipeline operations
 */
export interface ILidarProcessor {
    /**
     * Process raw point cloud data with specified configuration
     * @param rawData Buffer containing raw LiDAR data
     * @param config Processing configuration
     * @returns Processed point cloud data
     * @throws Error if processing time exceeds 50ms
     */
    processPointCloud(rawData: Buffer, config: ILidarConfig): Promise<IPointCloud>;

    /**
     * Validate scan results against quality thresholds
     * @param pointCloud Processed point cloud data
     * @returns Validation result with confidence score
     */
    validateScan(pointCloud: IPointCloud): Promise<{
        isValid: boolean;
        confidence: number;
        errors?: string[];
    }>;

    /**
     * Generate metadata for scan operation
     * @param pointCloud Processed point cloud
     * @param processingTime Processing duration
     * @returns Scan metadata
     */
    generateScanMetadata(
        pointCloud: IPointCloud,
        processingTime: number
    ): IScanMetadata;

    /**
     * Optimize scan rate based on current conditions
     * @param currentConfig Current LiDAR configuration
     * @param systemLoad Current system load
     * @returns Optimized scan rate
     */
    optimizeScanRate(
        currentConfig: ILidarConfig,
        systemLoad: number
    ): Promise<number>;

    /**
     * Calibrate LiDAR sensor
     * @returns Updated calibration data
     */
    calibrateSensor(): Promise<CalibrationData>;
}

// Global constants for LiDAR processing constraints
export const MAX_SCAN_RATE = 30;                  // Maximum scan rate in Hz
export const MIN_RESOLUTION = 0.01;               // Minimum resolution in cm
export const MAX_RANGE = 5.0;                     // Maximum range in meters
export const MAX_PROCESSING_TIME = 50;            // Maximum processing time in ms
export const MIN_POINT_DENSITY = 1000;            // Minimum points per cubic meter
export const MAX_ERROR_RATE = 0.001;              // Maximum acceptable error rate
export const MIN_CONFIDENCE_THRESHOLD = 0.95;      // Minimum confidence threshold