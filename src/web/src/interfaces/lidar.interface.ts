import { Buffer } from 'buffer'; // latest
import { 
    LIDAR_SCAN_SETTINGS, 
    LIDAR_VISUALIZATION 
} from '../constants/lidar.constants';

/**
 * Enum defining LiDAR processing modes affecting performance and quality tradeoffs
 */
export enum ProcessingMode {
    REAL_TIME = 'REAL_TIME',    // Optimized for lowest latency
    QUALITY = 'QUALITY',        // Optimized for highest accuracy
    POWER_SAVE = 'POWER_SAVE'   // Optimized for power efficiency
}

/**
 * Enum defining scan quality levels
 */
export enum ScanQuality {
    HIGH = 'HIGH',       // >95% confidence
    MEDIUM = 'MEDIUM',   // 80-95% confidence
    LOW = 'LOW'         // <80% confidence
}

/**
 * Interface for LiDAR scan configuration
 */
export interface ILidarScanConfig {
    scanFrequency: typeof LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY;
    resolution: typeof LIDAR_SCAN_SETTINGS.RESOLUTION;
    range: typeof LIDAR_SCAN_SETTINGS.MAX_RANGE;
    adaptiveQuality: boolean;
    processingMode: ProcessingMode;
}

/**
 * Interface for 3D point data with quality metrics
 */
export interface IPoint3D {
    x: number;
    y: number;
    z: number;
    intensity: number;      // Signal intensity (0-1)
    confidence: number;     // Measurement confidence (0-1)
}

/**
 * Interface for scan metadata
 */
export interface IScanMetadata {
    deviceId: string;
    timestamp: number;
    scanDuration: number;   // Duration in milliseconds
    scannerHealth: number;  // Health status (0-1)
    environmentalFactors: {
        temperature: number;
        humidity: number;
        ambientLight: number;
    };
}

/**
 * Interface for bounding box dimensions
 */
export interface IBoundingBox {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    minZ: number;
    maxZ: number;
}

/**
 * Comprehensive interface for point cloud data
 */
export interface IPointCloud {
    points: IPoint3D[];
    timestamp: number;
    quality: ScanQuality;
    rawData: Buffer;
    metadata: IScanMetadata;
    boundingBox: IBoundingBox;
}

/**
 * Interface for visualization settings
 */
export interface IVisualizationSettings {
    pointSize: typeof LIDAR_VISUALIZATION.POINT_SIZE;
    colorScheme: typeof LIDAR_VISUALIZATION.COLOR_SCHEME;
    opacity: typeof LIDAR_VISUALIZATION.OPACITY;
    highlightIntensity: boolean;
    showConfidenceMap: boolean;
}

/**
 * Interface for LiDAR performance metrics
 */
export interface ILidarPerformanceMetrics {
    processingLatency: number;      // Processing time in milliseconds
    pointsPerSecond: number;        // Points processed per second
    memoryUsage: number;            // Memory usage in bytes
    bufferUtilization: number;      // Buffer utilization percentage (0-1)
    qualityMetrics: {
        confidenceAverage: number;   // Average confidence score (0-1)
        noiseLevels: number;        // Estimated noise levels (0-1)
        outlierPercentage: number;  // Percentage of detected outliers
    };
    systemLoad: {
        cpu: number;                // CPU utilization (0-1)
        gpu: number;                // GPU utilization (0-1)
        memory: number;             // Memory utilization (0-1)
    };
}

/**
 * Interface for scan processing pipeline configuration
 */
export interface IProcessingPipelineConfig {
    outlierRemoval: {
        enabled: boolean;
        threshold: number;
    };
    downsampling: {
        enabled: boolean;
        targetPointCount: number;
    };
    smoothing: {
        enabled: boolean;
        kernelSize: number;
    };
    classification: {
        enabled: boolean;
        confidenceThreshold: number;
    };
}