import { Buffer } from 'buffer'; // version: latest
import {
  ILidarConfig,
  IPointCloud,
  IScanMetadata,
  ILidarProcessor,
  ProcessingMode,
  PowerMode,
  ScanQuality,
  MAX_SCAN_RATE,
  MIN_RESOLUTION,
  MAX_RANGE,
  MAX_PROCESSING_TIME,
  MIN_POINT_DENSITY,
  MAX_ERROR_RATE,
  MIN_CONFIDENCE_THRESHOLD
} from '../interfaces/lidar.interface';

// Global constants for system constraints
export const MAX_MEMORY_USAGE = 512 * 1024 * 1024; // 512MB max memory usage
export const MIN_POINTS_PER_SCAN = 100000; // Minimum points per scan
export const PROCESSING_TIMEOUT = MAX_PROCESSING_TIME; // 50ms processing timeout

/**
 * Performance mode options for LiDAR processing
 */
export enum PerformanceMode {
  ULTRA = 'ULTRA',         // Maximum performance, highest power consumption
  BALANCED = 'BALANCED',   // Balanced performance and power usage
  EFFICIENT = 'EFFICIENT', // Power-efficient mode with reduced performance
  ADAPTIVE = 'ADAPTIVE'    // Dynamically adjusts based on conditions
}

/**
 * Hardware status information for validation
 */
export interface HardwareStatus {
  temperature: number;
  powerDraw: number;
  memoryUsage: number;
  processingLoad: number;
  sensorHealth: number;
}

/**
 * Detailed processing metrics
 */
export interface ProcessingMetrics {
  processingTime: number;
  pointCount: number;
  memoryUsage: number;
  powerConsumption: number;
  qualityScore: number;
  confidence: number;
}

/**
 * Validation error structure
 */
export interface ValidationError {
  code: string;
  message: string;
  severity: 'critical' | 'error';
  component: string;
  timestamp: number;
}

/**
 * Validation warning structure
 */
export interface ValidationWarning {
  code: string;
  message: string;
  component: string;
  timestamp: number;
  recommendation?: string;
}

/**
 * Enhanced point cloud data structure with processing metrics
 */
export type PointCloudData = {
  rawData: Buffer;
  processedData: IPointCloud;
  processingMetrics: ProcessingMetrics;
  quality: ScanQuality;
  timestamp: number;
};

/**
 * Extended LiDAR configuration options
 */
export type LidarConfigOptions = {
  config: Partial<ILidarConfig>;
  performanceMode: PerformanceMode;
  adaptiveSettings?: {
    powerThreshold: number;
    qualityThreshold: number;
    thermalLimit: number;
  };
};

/**
 * Comprehensive processing result type
 */
export type ProcessingResult = {
  pointCloud: IPointCloud;
  metadata: IScanMetadata;
  performance: ProcessingMetrics;
  validation: ValidationResult;
};

/**
 * Detailed validation result type
 */
export type ValidationResult = {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  hardwareStatus: HardwareStatus;
  qualityMetrics: {
    resolution: number;
    accuracy: number;
    precision: number;
    coverage: number;
  };
};

/**
 * Type guard to validate scan rate against specifications
 */
export function isValidScanRate(rate: number): rate is number {
  return (
    typeof rate === 'number' &&
    rate > 0 &&
    rate <= MAX_SCAN_RATE &&
    Number.isFinite(rate)
  );
}

/**
 * Type guard to validate scan resolution
 */
export function isValidResolution(resolution: number): resolution is number {
  return (
    typeof resolution === 'number' &&
    resolution >= MIN_RESOLUTION &&
    Number.isFinite(resolution)
  );
}

/**
 * Validates processing metrics against requirements
 */
export function validateProcessingMetrics(metrics: ProcessingMetrics): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  
  // Validate processing time
  if (metrics.processingTime > MAX_PROCESSING_TIME) {
    errors.push({
      code: 'PROCESSING_TIME_EXCEEDED',
      message: `Processing time ${metrics.processingTime}ms exceeds maximum ${MAX_PROCESSING_TIME}ms`,
      severity: 'critical',
      component: 'processor',
      timestamp: Date.now()
    });
  }

  // Validate memory usage
  if (metrics.memoryUsage > MAX_MEMORY_USAGE) {
    errors.push({
      code: 'MEMORY_LIMIT_EXCEEDED',
      message: 'Memory usage exceeds maximum allowed limit',
      severity: 'error',
      component: 'memory',
      timestamp: Date.now()
    });
  }

  // Validate point count
  if (metrics.pointCount < MIN_POINTS_PER_SCAN) {
    warnings.push({
      code: 'LOW_POINT_DENSITY',
      message: 'Point cloud density below recommended threshold',
      component: 'scanner',
      timestamp: Date.now(),
      recommendation: 'Consider adjusting scan resolution or range'
    });
  }

  // Generate hardware status
  const hardwareStatus: HardwareStatus = {
    temperature: 0, // Placeholder for actual hardware monitoring
    powerDraw: metrics.powerConsumption,
    memoryUsage: metrics.memoryUsage,
    processingLoad: (metrics.processingTime / MAX_PROCESSING_TIME) * 100,
    sensorHealth: metrics.confidence
  };

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    hardwareStatus,
    qualityMetrics: {
      resolution: metrics.qualityScore,
      accuracy: metrics.confidence,
      precision: metrics.qualityScore * metrics.confidence,
      coverage: (metrics.pointCount / MIN_POINTS_PER_SCAN) * 100
    }
  };
}