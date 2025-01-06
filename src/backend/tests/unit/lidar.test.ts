import { describe, it, expect, jest, beforeEach, afterEach, beforeAll, afterAll } from '@jest/globals'; // version: ^29.0.0
import { performance } from 'perf_hooks'; // version: node built-in
import { PerformanceMonitor } from '@performance-monitor'; // version: ^2.0.0

import { LidarProcessor } from '../../src/core/lidar/LidarProcessor';
import { PointCloudGenerator } from '../../src/core/lidar/PointCloudGenerator';
import {
    ILidarConfig,
    ProcessingMode,
    ScanQuality,
    PowerMode,
    IPointCloud
} from '../../interfaces/lidar.interface';

// Global test constants
const SCAN_RATE_HZ = 30;
const MIN_RESOLUTION_CM = 0.01;
const MAX_RANGE_M = 5.0;
const PROCESSING_TIMEOUT_MS = 50;
const GPU_MEMORY_THRESHOLD = 2048;
const MIN_POINT_DENSITY = 1000;

// Mock test data and configurations
const mockLidarConfig: ILidarConfig = {
    scanRate: SCAN_RATE_HZ,
    resolution: MIN_RESOLUTION_CM,
    range: MAX_RANGE_M,
    processingMode: ProcessingMode.REAL_TIME,
    powerMode: PowerMode.BALANCED,
    calibrationData: {
        offsetX: 0,
        offsetY: 0,
        offsetZ: 0,
        rotationMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        distortionParams: [0, 0, 0],
        timestamp: Date.now()
    }
};

// Mock performance monitor
const mockPerformanceMonitor = {
    startMeasurement: jest.fn(),
    endMeasurement: jest.fn(),
    getMetrics: jest.fn()
};

describe('LiDAR Processing Pipeline Tests', () => {
    let lidarProcessor: LidarProcessor;
    let pointCloudGenerator: PointCloudGenerator;
    let performanceMonitor: PerformanceMonitor;

    beforeAll(async () => {
        performanceMonitor = new PerformanceMonitor();
        await performanceMonitor.initialize();
    });

    beforeEach(() => {
        // Reset mocks and create fresh instances
        jest.clearAllMocks();
        pointCloudGenerator = new PointCloudGenerator(mockLidarConfig);
        lidarProcessor = new LidarProcessor(mockLidarConfig, pointCloudGenerator);
    });

    afterEach(async () => {
        await performanceMonitor.clearMetrics();
    });

    afterAll(async () => {
        await performanceMonitor.shutdown();
    });

    describe('Performance Requirements', () => {
        it('should maintain 30Hz scan rate under load', async () => {
            const iterations = 100;
            const results: number[] = [];

            for (let i = 0; i < iterations; i++) {
                const startTime = performance.now();
                await lidarProcessor.processPointCloud(Buffer.alloc(1024));
                const endTime = performance.now();
                results.push(endTime - startTime);
            }

            const averageTime = results.reduce((a, b) => a + b) / iterations;
            const scanRate = 1000 / averageTime;

            expect(scanRate).toBeGreaterThanOrEqual(SCAN_RATE_HZ);
            expect(Math.max(...results)).toBeLessThanOrEqual(PROCESSING_TIMEOUT_MS);
        });

        it('should process point clouds within 50ms latency', async () => {
            const startTime = performance.now();
            await lidarProcessor.processPointCloud(Buffer.alloc(1024));
            const processingTime = performance.now() - startTime;

            expect(processingTime).toBeLessThanOrEqual(PROCESSING_TIMEOUT_MS);
        });

        it('should optimize GPU memory usage', async () => {
            const memoryMetrics = await performanceMonitor.getMetrics('gpu_memory');
            
            expect(memoryMetrics.peakUsage).toBeLessThanOrEqual(GPU_MEMORY_THRESHOLD);
            expect(memoryMetrics.averageUsage).toBeLessThanOrEqual(GPU_MEMORY_THRESHOLD * 0.8);
        });

        it('should handle thermal throttling gracefully', async () => {
            // Simulate high temperature condition
            jest.spyOn(performanceMonitor, 'getGPUTemperature').mockReturnValue(85);

            const result = await lidarProcessor.processPointCloud(Buffer.alloc(1024));
            
            expect(result.metadata.quality).toBe(ScanQuality.MEDIUM);
            expect(result.metadata.processingTime).toBeLessThanOrEqual(PROCESSING_TIMEOUT_MS);
        });
    });

    describe('Quality Metrics', () => {
        it('should achieve 0.01cm resolution', async () => {
            const result = await pointCloudGenerator.generatePointCloud(Buffer.alloc(1024));
            const validation = await pointCloudGenerator.validatePointCloud(result.pointCloud);

            expect(validation.qualityMetrics.resolution).toBeLessThanOrEqual(MIN_RESOLUTION_CM);
            expect(validation.isValid).toBe(true);
        });

        it('should maintain accuracy up to 5 meters', async () => {
            const testRanges = [1, 2, 3, 4, 5];
            
            for (const range of testRanges) {
                const config = { ...mockLidarConfig, range };
                const generator = new PointCloudGenerator(config);
                const result = await generator.generatePointCloud(Buffer.alloc(1024));
                
                expect(result.validation.qualityMetrics.accuracy).toBeGreaterThanOrEqual(0.95);
            }
        });

        it('should meet minimum point density requirements', async () => {
            const result = await pointCloudGenerator.generatePointCloud(Buffer.alloc(1024));
            
            expect(result.pointCloud.density).toBeGreaterThanOrEqual(MIN_POINT_DENSITY);
            expect(result.validation.qualityMetrics.coverage).toBeGreaterThanOrEqual(95);
        });

        it('should effectively reduce noise', async () => {
            const noisyData = Buffer.alloc(1024).fill(Math.random());
            const result = await lidarProcessor.processPointCloud(noisyData);
            
            expect(result.metadata.errorRate).toBeLessThanOrEqual(0.05);
            expect(result.metadata.quality).not.toBe(ScanQuality.LOW);
        });
    });

    describe('Error Handling', () => {
        it('should handle hardware failures', async () => {
            // Simulate hardware error
            jest.spyOn(pointCloudGenerator, 'generatePointCloud')
                .mockRejectedValueOnce(new Error('Hardware failure'));

            await expect(lidarProcessor.processPointCloud(Buffer.alloc(1024)))
                .rejects.toThrow('Hardware failure');
        });

        it('should recover from processing errors', async () => {
            let errorCount = 0;
            const iterations = 5;

            for (let i = 0; i < iterations; i++) {
                try {
                    if (i === 2) { // Simulate error on third iteration
                        jest.spyOn(pointCloudGenerator, 'generatePointCloud')
                            .mockRejectedValueOnce(new Error('Processing error'));
                    }
                    await lidarProcessor.processPointCloud(Buffer.alloc(1024));
                } catch (error) {
                    errorCount++;
                }
            }

            expect(errorCount).toBe(1);
            const finalResult = await lidarProcessor.processPointCloud(Buffer.alloc(1024));
            expect(finalResult.metadata.quality).not.toBe(ScanQuality.LOW);
        });

        it('should manage resource exhaustion', async () => {
            // Simulate low memory condition
            jest.spyOn(performanceMonitor, 'getMemoryUsage').mockReturnValue(0.95);

            const result = await lidarProcessor.processPointCloud(Buffer.alloc(1024));
            
            expect(result.metadata.processingTime).toBeLessThanOrEqual(PROCESSING_TIMEOUT_MS);
            expect(result.metadata.quality).toBe(ScanQuality.MEDIUM);
        });

        it('should validate corrupt scan data', async () => {
            const corruptData = Buffer.from('corrupt data');
            
            await expect(lidarProcessor.processPointCloud(corruptData))
                .rejects.toThrow(/Invalid scan data/);
        });
    });

    describe('Configuration Validation', () => {
        it('should reject invalid scan rates', () => {
            const invalidConfig = { ...mockLidarConfig, scanRate: 35 };
            
            expect(() => new LidarProcessor(invalidConfig, pointCloudGenerator))
                .toThrow(/Invalid scan rate/);
        });

        it('should reject invalid resolutions', () => {
            const invalidConfig = { ...mockLidarConfig, resolution: 0.001 };
            
            expect(() => new LidarProcessor(invalidConfig, pointCloudGenerator))
                .toThrow(/Invalid resolution/);
        });

        it('should reject invalid ranges', () => {
            const invalidConfig = { ...mockLidarConfig, range: 6.0 };
            
            expect(() => new LidarProcessor(invalidConfig, pointCloudGenerator))
                .toThrow(/Invalid range/);
        });
    });
});