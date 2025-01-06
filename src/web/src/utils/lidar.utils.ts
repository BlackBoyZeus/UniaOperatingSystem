import { Buffer } from 'buffer'; // ^6.0.3
import { Point3D, PointCloudData, ScanQuality } from '../types/lidar.types';
import { LIDAR_SCAN_SETTINGS } from '../constants/lidar.constants';

// Global constants for performance optimization and quality thresholds
const MIN_POINT_DENSITY = 1000;
const MAX_POINT_DENSITY = 1000000;
const QUALITY_THRESHOLD_HIGH = 0.9;
const QUALITY_THRESHOLD_MEDIUM = 0.6;
const MAX_PROCESSING_TIME = 50; // ms, per technical spec
const MEMORY_POOL_SIZE = 1024 * 1024 * 10; // 10MB memory pool

/**
 * Memory pool for optimized point cloud data handling
 */
const memoryPool = new ArrayBuffer(MEMORY_POOL_SIZE);
let memoryOffset = 0;

/**
 * Validates LiDAR scan configuration parameters against system constraints
 * @param config - Scan configuration object
 * @returns boolean indicating if configuration is valid
 */
export function validateScanConfig(config: {
    frequency: number;
    resolution: number;
    range: number;
}): boolean {
    try {
        // Validate scan frequency (0-30Hz)
        if (config.frequency < 0 || config.frequency > LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY) {
            return false;
        }

        // Validate resolution (>= 0.01cm)
        if (config.resolution < LIDAR_SCAN_SETTINGS.RESOLUTION) {
            return false;
        }

        // Validate range (<= 5.0m)
        if (config.range > LIDAR_SCAN_SETTINGS.MAX_RANGE) {
            return false;
        }

        // Validate memory constraints
        const estimatedMemory = (config.range / config.resolution) * 3 * 4; // xyz * 4 bytes
        if (estimatedMemory > MEMORY_POOL_SIZE) {
            return false;
        }

        return true;
    } catch (error) {
        console.error('Scan config validation error:', error);
        return false;
    }
}

/**
 * Converts raw binary point cloud data into structured Point3D array
 * @param rawData - Binary buffer containing point cloud data
 * @returns Array of Point3D objects
 */
export function parsePointCloudBuffer(rawData: Buffer): Point3D[] {
    try {
        // Reset memory pool if near capacity
        if (memoryOffset > MEMORY_POOL_SIZE * 0.9) {
            memoryOffset = 0;
        }

        const view = new DataView(memoryPool);
        const points: Point3D[] = [];
        let offset = 0;

        while (offset < rawData.length) {
            // Read coordinates (12 bytes per point: x,y,z as float32)
            const x = rawData.readFloatLE(offset);
            const y = rawData.readFloatLE(offset + 4);
            const z = rawData.readFloatLE(offset + 8);

            // Validate point coordinates
            if (isFinite(x) && isFinite(y) && isFinite(z)) {
                points.push({ x, y, z });
                
                // Store in memory pool
                view.setFloat32(memoryOffset, x);
                view.setFloat32(memoryOffset + 4, y);
                view.setFloat32(memoryOffset + 8, z);
                memoryOffset += 12;
            }

            offset += 12;
        }

        return points;
    } catch (error) {
        console.error('Point cloud parsing error:', error);
        return [];
    }
}

/**
 * Optimizes point cloud data for visualization with enhanced algorithms
 * @param points - Array of Point3D objects
 * @param targetDensity - Desired point density
 * @returns Optimized array of Point3D objects
 */
export function optimizePointCloud(points: Point3D[], targetDensity: number): Point3D[] {
    try {
        // Validate target density
        const validDensity = Math.max(
            MIN_POINT_DENSITY,
            Math.min(targetDensity, MAX_POINT_DENSITY)
        );

        // Calculate current density
        const currentDensity = points.length;
        
        if (currentDensity <= validDensity) {
            return points;
        }

        // Calculate sampling rate
        const samplingRate = validDensity / currentDensity;
        
        // Apply spatial partitioning and downsampling
        const optimizedPoints: Point3D[] = [];
        const gridSize = Math.cbrt(validDensity);
        const grid: Map<string, Point3D> = new Map();

        for (const point of points) {
            const gridX = Math.floor(point.x * gridSize);
            const gridY = Math.floor(point.y * gridSize);
            const gridZ = Math.floor(point.z * gridSize);
            const key = `${gridX},${gridY},${gridZ}`;

            if (Math.random() < samplingRate && !grid.has(key)) {
                grid.set(key, point);
                optimizedPoints.push(point);
            }
        }

        return optimizedPoints;
    } catch (error) {
        console.error('Point cloud optimization error:', error);
        return points;
    }
}

/**
 * Calculates detailed performance metrics for scan processing
 * @param startTime - Processing start timestamp
 * @param pointCount - Number of points processed
 * @returns Object containing processing metrics
 */
export function calculateProcessingMetrics(startTime: number, pointCount: number): {
    processingTime: number;
    pointsPerSecond: number;
    quality: ScanQuality;
    withinLatencyTarget: boolean;
} {
    try {
        const endTime = performance.now();
        const processingTime = endTime - startTime;
        const pointsPerSecond = (pointCount / processingTime) * 1000;

        // Calculate quality based on processing metrics
        let quality: ScanQuality;
        const qualityScore = Math.min(
            pointsPerSecond / MAX_POINT_DENSITY,
            processingTime / MAX_PROCESSING_TIME
        );

        if (qualityScore >= QUALITY_THRESHOLD_HIGH) {
            quality = ScanQuality.HIGH;
        } else if (qualityScore >= QUALITY_THRESHOLD_MEDIUM) {
            quality = ScanQuality.MEDIUM;
        } else {
            quality = ScanQuality.LOW;
        }

        return {
            processingTime,
            pointsPerSecond,
            quality,
            withinLatencyTarget: processingTime <= MAX_PROCESSING_TIME
        };
    } catch (error) {
        console.error('Processing metrics calculation error:', error);
        return {
            processingTime: 0,
            pointsPerSecond: 0,
            quality: ScanQuality.LOW,
            withinLatencyTarget: false
        };
    }
}