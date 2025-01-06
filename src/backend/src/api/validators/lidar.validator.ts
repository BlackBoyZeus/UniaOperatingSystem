import { z } from 'zod'; // v3.22.2
import {
    ILidarConfig,
    IPointCloud,
    IScanMetadata,
    ScanQuality,
    ProcessingMode,
    PowerMode,
    MAX_SCAN_RATE,
    MIN_RESOLUTION,
    MAX_RANGE,
    MAX_PROCESSING_TIME
} from '../../interfaces/lidar.interface';
import { validateLidarConfig } from '../../utils/validation.utils';

/**
 * Zod schema for validating LiDAR configuration
 * Ensures compliance with TALD UNIA platform's real-time scanning requirements:
 * - 30Hz continuous scanning
 * - 0.01cm resolution
 * - 5-meter effective range
 * - ≤50ms processing latency
 */
export const lidarConfigSchema = z.object({
    scanRate: z.number()
        .min(1)
        .max(MAX_SCAN_RATE)
        .refine(val => val <= 30, 'Scan rate must not exceed 30Hz'),

    resolution: z.number()
        .min(MIN_RESOLUTION)
        .refine(val => val >= 0.01, 'Resolution must be at least 0.01cm'),

    range: z.number()
        .min(0.1)
        .max(MAX_RANGE)
        .refine(val => val <= 5.0, 'Range must not exceed 5.0 meters'),

    processingMode: z.nativeEnum(ProcessingMode)
        .refine(mode => 
            mode === ProcessingMode.REAL_TIME ? true : 
            mode === ProcessingMode.HIGH_QUALITY ? true :
            mode === ProcessingMode.POWER_SAVE,
            'Invalid processing mode'
        ),

    powerMode: z.nativeEnum(PowerMode)
        .refine(mode => 
            mode === PowerMode.PERFORMANCE ? true :
            mode === PowerMode.BALANCED ? true :
            mode === PowerMode.EFFICIENCY,
            'Invalid power mode'
        ),

    calibrationData: z.object({
        offsetX: z.number(),
        offsetY: z.number(),
        offsetZ: z.number(),
        rotationMatrix: z.array(z.array(z.number())).length(3)
            .refine(matrix => matrix.every(row => row.length === 3), 
                'Rotation matrix must be 3x3'),
        distortionParams: z.array(z.number()),
        timestamp: z.number()
            .refine(ts => ts <= Date.now(), 'Calibration timestamp cannot be in the future')
    })
});

/**
 * Zod schema for validating point cloud data structure
 * Ensures data quality and format compliance for real-time processing
 */
export const pointCloudSchema = z.object({
    points: z.instanceof(Buffer)
        .refine(buffer => buffer.length > 0, 'Point cloud buffer cannot be empty'),

    timestamp: z.number()
        .refine(ts => ts <= Date.now(), 'Timestamp cannot be in the future')
        .refine(ts => Date.now() - ts <= 1000, 'Point cloud data too old'),

    quality: z.nativeEnum(ScanQuality)
        .refine(quality => 
            quality === ScanQuality.HIGH ? true :
            quality === ScanQuality.MEDIUM ? true :
            quality === ScanQuality.LOW,
            'Invalid scan quality'
        ),

    density: z.number()
        .min(1000)
        .refine(val => val >= 1000, 'Minimum point density of 1000 points/m³ required'),

    confidence: z.number()
        .min(0)
        .max(1)
        .refine(val => val >= 0.95, 'Minimum confidence score of 0.95 required')
});

/**
 * Zod schema for validating scan metadata
 * Ensures compliance with performance requirements and quality metrics
 */
export const scanMetadataSchema = z.object({
    scanId: z.string().uuid(),

    timestamp: z.number()
        .refine(ts => ts <= Date.now(), 'Timestamp cannot be in the future'),

    processingTime: z.number()
        .max(MAX_PROCESSING_TIME)
        .refine(time => time <= 50, 'Processing time must not exceed 50ms'),

    quality: z.nativeEnum(ScanQuality)
        .refine(quality => 
            quality === ScanQuality.HIGH ? true :
            quality === ScanQuality.MEDIUM ? true :
            quality === ScanQuality.LOW,
            'Invalid scan quality'
        ),

    errorRate: z.number()
        .min(0)
        .max(1)
        .refine(rate => rate <= 0.001, 'Error rate must not exceed 0.1%'),

    powerConsumption: z.number()
        .min(0)
        .refine(power => power <= 5.0, 'Power consumption must not exceed 5.0W')
});

/**
 * Validates LiDAR configuration against hardware capabilities and performance requirements
 * @param config LiDAR configuration to validate
 * @returns Promise resolving to true if configuration is valid
 */
export async function validateLidarConfiguration(config: ILidarConfig): Promise<boolean> {
    try {
        // Validate schema
        await lidarConfigSchema.parseAsync(config);

        // Perform hardware-specific validation
        const validationResult = await validateLidarConfig(
            config,
            config.calibrationData,
            {
                scanId: '',
                timestamp: Date.now(),
                processingTime: 0,
                quality: ScanQuality.HIGH,
                errorRate: 0,
                powerConsumption: 0
            }
        );

        return validationResult.valid;
    } catch (error) {
        console.error('LiDAR configuration validation failed:', error);
        return false;
    }
}

/**
 * Validates point cloud data structure, format, and quality metrics
 * @param pointCloud Point cloud data to validate
 * @returns Promise resolving to true if point cloud data is valid
 */
export async function validatePointCloudData(pointCloud: IPointCloud): Promise<boolean> {
    try {
        await pointCloudSchema.parseAsync(pointCloud);
        return true;
    } catch (error) {
        console.error('Point cloud validation failed:', error);
        return false;
    }
}

/**
 * Validates scan metadata including processing performance and quality metrics
 * @param metadata Scan metadata to validate
 * @returns Promise resolving to true if metadata is valid
 */
export async function validateScanMetadata(metadata: IScanMetadata): Promise<boolean> {
    try {
        await scanMetadataSchema.parseAsync(metadata);
        return true;
    } catch (error) {
        console.error('Scan metadata validation failed:', error);
        return false;
    }
}