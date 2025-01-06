import { renderHook, act } from '@testing-library/react-hooks';
import { waitFor } from '@testing-library/react';
import { performance } from 'jest-performance';
import { server } from '../mocks/server';
import { useLidar } from '../../src/hooks/useLidar';
import { LIDAR_SCAN_SETTINGS } from '../../src/constants/lidar.constants';
import { ScanQuality } from '../../src/types/lidar.types';

// Test constants
const TEST_TIMEOUT = 10000;
const MOCK_POINT_CLOUD_SIZE = 1_000_000;
const PERFORMANCE_TEST_DURATION = 5000;
const LATENCY_THRESHOLD = 50;
const SCAN_FREQUENCY = 30;

describe('useLidar hook', () => {
    // Setup and cleanup
    beforeEach(() => {
        server.resetHandlers();
        performance.mark('test-start');
    });

    afterEach(() => {
        performance.clearMarks();
        performance.clearMeasures();
    });

    it('should initialize with correct default state', () => {
        const { result } = renderHook(() => useLidar());

        expect(result.current.scanState).toEqual({
            isActive: false,
            currentScan: null,
            metadata: null
        });

        expect(result.current.performanceMetrics).toEqual({
            processingLatency: 0,
            pointsProcessed: 0,
            quality: ScanQuality.HIGH,
            fleetSyncLatency: 0
        });
    });

    it('should validate scan frequency and resolution requirements', async () => {
        const { result } = renderHook(() => useLidar());
        const scanIntervals: number[] = [];
        let lastScanTime = performance.now();

        // Start scanning
        await act(async () => {
            await result.current.startScanning();
        });

        // Monitor scan frequency for 1 second
        await waitFor(
            () => {
                const currentTime = performance.now();
                scanIntervals.push(currentTime - lastScanTime);
                lastScanTime = currentTime;
                return scanIntervals.length >= SCAN_FREQUENCY;
            },
            { timeout: TEST_TIMEOUT }
        );

        // Calculate average scan interval
        const avgInterval = scanIntervals.reduce((a, b) => a + b) / scanIntervals.length;
        const actualFrequency = 1000 / avgInterval;

        // Verify scanning requirements
        expect(actualFrequency).toBeCloseTo(LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY, 1);
        expect(result.current.scanState.isActive).toBe(true);
        expect(result.current.performanceMetrics.quality).toBe(ScanQuality.HIGH);
    });

    it('should verify point cloud processing performance', async () => {
        const { result } = renderHook(() => useLidar());
        const processingTimes: number[] = [];

        await act(async () => {
            await result.current.startScanning();
        });

        // Monitor processing latency
        const startTime = performance.now();
        while (performance.now() - startTime < PERFORMANCE_TEST_DURATION) {
            if (result.current.performanceMetrics.processingLatency > 0) {
                processingTimes.push(result.current.performanceMetrics.processingLatency);
            }
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        // Calculate performance metrics
        const avgProcessingTime = processingTimes.reduce((a, b) => a + b) / processingTimes.length;
        const maxProcessingTime = Math.max(...processingTimes);

        // Verify performance requirements
        expect(avgProcessingTime).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        expect(maxProcessingTime).toBeLessThanOrEqual(LATENCY_THRESHOLD * 1.2); // Allow 20% margin
        expect(result.current.performanceMetrics.pointsProcessed).toBeGreaterThan(0);
    });

    it('should handle visualization updates efficiently', async () => {
        const { result } = renderHook(() => useLidar());
        const visualizationConfig = {
            pointSize: 2,
            colorScheme: 'rainbow',
            opacity: 0.8
        };

        await act(async () => {
            await result.current.startScanning();
            await result.current.updateVisualization(visualizationConfig);
        });

        // Verify visualization update performance
        expect(result.current.performanceMetrics.processingLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        expect(result.current.scanState.isActive).toBe(true);
    });

    it('should maintain performance during fleet synchronization', async () => {
        const { result } = renderHook(() => useLidar());
        const syncLatencies: number[] = [];

        await act(async () => {
            await result.current.startScanning();
        });

        // Monitor fleet sync latency
        const startTime = performance.now();
        while (performance.now() - startTime < PERFORMANCE_TEST_DURATION) {
            if (result.current.performanceMetrics.fleetSyncLatency > 0) {
                syncLatencies.push(result.current.performanceMetrics.fleetSyncLatency);
            }
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        const avgSyncLatency = syncLatencies.reduce((a, b) => a + b) / syncLatencies.length;

        // Verify fleet sync performance
        expect(avgSyncLatency).toBeLessThanOrEqual(LATENCY_THRESHOLD);
        expect(result.current.fleetState.isConnected).toBe(true);
    });

    it('should handle cleanup and resource management properly', async () => {
        const { result, unmount } = renderHook(() => useLidar());

        await act(async () => {
            await result.current.startScanning();
        });

        // Verify active scanning
        expect(result.current.scanState.isActive).toBe(true);

        // Unmount and verify cleanup
        unmount();

        await waitFor(() => {
            expect(result.current.scanState.isActive).toBe(false);
            expect(result.current.currentPointCloud).toBeNull();
        });
    });

    it('should maintain consistent scan quality under load', async () => {
        const { result } = renderHook(() => useLidar());
        const qualityMeasurements: ScanQuality[] = [];

        await act(async () => {
            await result.current.startScanning();
        });

        // Generate high load
        const startTime = performance.now();
        while (performance.now() - startTime < PERFORMANCE_TEST_DURATION) {
            qualityMeasurements.push(result.current.performanceMetrics.quality);
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        // Calculate quality statistics
        const highQualityRatio = qualityMeasurements.filter(q => q === ScanQuality.HIGH).length / 
            qualityMeasurements.length;

        // Verify quality requirements
        expect(highQualityRatio).toBeGreaterThanOrEqual(0.9); // 90% high quality scans
        expect(result.current.performanceMetrics.quality).toBe(ScanQuality.HIGH);
    });
});