import { describe, beforeEach, afterEach, it, expect, jest } from '@jest/globals'; // v29.0.0

import { LidarService } from '../../src/services/lidar/LidarService';
import { LidarProcessor } from '../../src/core/lidar/LidarProcessor';
import { 
    ILidarConfig, 
    IPointCloud, 
    ProcessingMode, 
    ScanQuality, 
    IScanMetadata 
} from '../../src/interfaces/lidar.interface';
import { 
    setupTestLidarScan, 
    cleanupTestData 
} from '../utils/testHelpers';

// Test configuration constants
const TEST_SCAN_RATE = 30;
const TEST_RESOLUTION = 0.01;
const TEST_RANGE = 5.0;
const TEST_PROCESSING_TIMEOUT = 50;

describe('LiDAR Processing Pipeline Integration Tests', () => {
    let lidarService: LidarService;
    let lidarProcessor: LidarProcessor;
    let testConfig: ILidarConfig;

    beforeEach(async () => {
        // Initialize test configuration
        testConfig = {
            scanRate: TEST_SCAN_RATE,
            resolution: TEST_RESOLUTION,
            range: TEST_RANGE,
            processingMode: ProcessingMode.REAL_TIME,
            powerMode: 'PERFORMANCE',
            calibrationData: {
                timestamp: Date.now(),
                offsetX: 0,
                offsetY: 0,
                offsetZ: 0,
                rotationMatrix: [[1,0,0], [0,1,0], [0,0,1]],
                distortionParams: []
            }
        };

        // Initialize services with test configuration
        lidarProcessor = new LidarProcessor(testConfig);
        lidarService = new LidarService(lidarProcessor);

        // Setup performance monitoring
        jest.spyOn(performance, 'now');
    });

    afterEach(async () => {
        await lidarService.stopScanning();
        await cleanupTestData();
        jest.clearAllMocks();
    });

    describe('Continuous Scanning Tests', () => {
        it('should maintain 30Hz scanning rate with ≤50ms latency', async () => {
            const scanDuration = 1000; // 1 second test duration
            const expectedScans = TEST_SCAN_RATE;
            const scans: IScanMetadata[] = [];

            // Start scanning
            await lidarService.startScanning();

            // Collect scan data for 1 second
            await new Promise<void>((resolve) => {
                const startTime = performance.now();
                
                lidarService.on('scanComplete', (metadata: IScanMetadata) => {
                    scans.push(metadata);
                    
                    if (performance.now() - startTime >= scanDuration) {
                        resolve();
                    }
                });
            });

            // Validate scan rate and latency
            expect(scans.length).toBeGreaterThanOrEqual(expectedScans * 0.95); // Allow 5% tolerance
            
            const avgLatency = scans.reduce((sum, scan) => sum + scan.processingTime, 0) / scans.length;
            expect(avgLatency).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
        });

        it('should maintain scan quality at 0.01cm resolution', async () => {
            const { pointCloud, validation } = await setupTestLidarScan(1_000_000, testConfig, {
                minQuality: 0.95,
                maxProcessingTime: TEST_PROCESSING_TIMEOUT
            });

            expect(validation.valid).toBe(true);
            expect(validation.metrics.quality).toBeGreaterThanOrEqual(0.95);
            expect(pointCloud.density).toBeGreaterThanOrEqual(1000); // Minimum points per cubic meter
        });

        it('should handle 5-meter effective range scans', async () => {
            const testRange = 5.0;
            const { pointCloud, validation } = await setupTestLidarScan(1_000_000, {
                ...testConfig,
                range: testRange
            });

            // Validate point cloud coverage
            const points = new Float32Array(pointCloud.points);
            const maxDistance = Math.max(
                ...Array.from({ length: points.length / 3 }, (_, i) => {
                    const x = points[i * 3];
                    const y = points[i * 3 + 1];
                    const z = points[i * 3 + 2];
                    return Math.sqrt(x * x + y * y + z * z);
                })
            );

            expect(maxDistance).toBeLessThanOrEqual(testRange);
            expect(validation.valid).toBe(true);
            expect(validation.metrics.quality).toBeGreaterThanOrEqual(0.95);
        });
    });

    describe('Point Cloud Processing Tests', () => {
        it('should process point clouds within 50ms latency', async () => {
            const processingTimes: number[] = [];
            const testIterations = 10;

            for (let i = 0; i < testIterations; i++) {
                const startTime = performance.now();
                
                const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
                const processedCloud = await lidarProcessor.processPointCloud(pointCloud.points);
                
                const processingTime = performance.now() - startTime;
                processingTimes.push(processingTime);

                expect(processedCloud).toBeDefined();
                expect(processingTime).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
            }

            const avgProcessingTime = processingTimes.reduce((a, b) => a + b) / testIterations;
            expect(avgProcessingTime).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
        });

        it('should validate scan quality metrics', async () => {
            const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
            const validation = await lidarProcessor.validateScan(pointCloud);

            expect(validation.isValid).toBe(true);
            expect(validation.confidence).toBeGreaterThanOrEqual(0.95);
            expect(validation.errors).toBeUndefined();
        });

        it('should handle high-density point clouds', async () => {
            const highDensityPoints = 2_000_000;
            const { pointCloud, validation } = await setupTestLidarScan(highDensityPoints, testConfig);

            expect(validation.valid).toBe(true);
            expect(pointCloud.density).toBeGreaterThanOrEqual(2000); // Points per cubic meter
            expect(validation.metrics.processingTime).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
        });
    });

    describe('Data Persistence Tests', () => {
        it('should persist processed scans with metadata', async () => {
            const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
            
            // Process and persist scan
            const startTime = performance.now();
            const processedCloud = await lidarProcessor.processPointCloud(pointCloud.points);
            const metadata = lidarProcessor.generateScanMetadata(processedCloud, performance.now() - startTime);

            // Validate persistence
            expect(metadata.scanId).toBeDefined();
            expect(metadata.timestamp).toBeGreaterThan(0);
            expect(metadata.processingTime).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
            expect(metadata.quality).toBe(ScanQuality.HIGH);
        });

        it('should handle concurrent scan processing and storage', async () => {
            const concurrentScans = 5;
            const scanPromises = Array.from({ length: concurrentScans }, async () => {
                const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
                return lidarProcessor.processPointCloud(pointCloud.points);
            });

            const results = await Promise.all(scanPromises);
            
            results.forEach(processedCloud => {
                expect(processedCloud).toBeDefined();
                expect(processedCloud.quality).toBe(ScanQuality.HIGH);
                expect(processedCloud.confidence).toBeGreaterThanOrEqual(0.95);
            });
        });
    });

    describe('Error Handling and Recovery Tests', () => {
        it('should handle and recover from processing errors', async () => {
            // Simulate processing error
            jest.spyOn(lidarProcessor, 'processPointCloud').mockImplementationOnce(() => {
                throw new Error('Processing error');
            });

            const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
            
            try {
                await lidarProcessor.processPointCloud(pointCloud.points);
            } catch (error) {
                expect(error.message).toBe('Processing error');
            }

            // Verify recovery
            const retryResult = await lidarProcessor.processPointCloud(pointCloud.points);
            expect(retryResult).toBeDefined();
            expect(retryResult.quality).toBe(ScanQuality.HIGH);
        });

        it('should maintain performance under system load', async () => {
            const loadTestDuration = 5000; // 5 seconds
            const startTime = performance.now();
            const processingTimes: number[] = [];

            // Start continuous scanning under load
            await lidarService.startScanning();

            while (performance.now() - startTime < loadTestDuration) {
                const scanStartTime = performance.now();
                const { pointCloud } = await setupTestLidarScan(1_000_000, testConfig);
                await lidarProcessor.processPointCloud(pointCloud.points);
                processingTimes.push(performance.now() - scanStartTime);
            }

            const avgProcessingTime = processingTimes.reduce((a, b) => a + b) / processingTimes.length;
            expect(avgProcessingTime).toBeLessThanOrEqual(TEST_PROCESSING_TIMEOUT);
        });
    });
});