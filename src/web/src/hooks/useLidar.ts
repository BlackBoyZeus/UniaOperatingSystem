import { useState, useEffect, useCallback, useMemo } from 'react'; // ^18.0.0
import { LidarService } from '../services/lidar.service';
import { FleetManager } from '../services/fleet.service';
import {
    Point3D,
    PointCloudData,
    ScanQuality,
    ScanConfig,
    ScanState,
    ScanMetadata
} from '../types/lidar.types';

// Constants for performance optimization
const SCAN_UPDATE_INTERVAL = 33.33; // 30Hz scanning rate
const CLEANUP_TIMEOUT = 100; // ms
const FLEET_SYNC_INTERVAL = 50; // ms
const PERFORMANCE_MONITOR_INTERVAL = 1000; // 1s

interface PerformanceMetrics {
    processingLatency: number;
    pointsProcessed: number;
    quality: ScanQuality;
    fleetSyncLatency: number;
}

interface LidarHookReturn {
    scanState: ScanState;
    fleetState: {
        isConnected: boolean;
        membersCount: number;
        syncStatus: string;
    };
    startScanning: () => Promise<void>;
    stopScanning: () => Promise<void>;
    updateVisualization: (config: any) => Promise<void>;
    currentPointCloud: PointCloudData | null;
    performanceMetrics: PerformanceMetrics;
}

/**
 * Custom hook for managing LiDAR functionality with fleet synchronization
 * Provides real-time access to LiDAR scanning with 30Hz update rate
 */
export function useLidar(): LidarHookReturn {
    // Core state management
    const [scanState, setScanState] = useState<ScanState>({
        isActive: false,
        currentScan: null,
        metadata: null
    });
    const [currentPointCloud, setCurrentPointCloud] = useState<PointCloudData | null>(null);
    const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
        processingLatency: 0,
        pointsProcessed: 0,
        quality: ScanQuality.HIGH,
        fleetSyncLatency: 0
    });
    const [fleetState, setFleetState] = useState({
        isConnected: false,
        membersCount: 0,
        syncStatus: 'idle'
    });

    // Service instances
    const lidarService = useMemo(() => new LidarService({
        frequency: 30,
        resolution: 0.01,
        range: 5.0
    }), []);

    const fleetManager = useMemo(() => new FleetManager(), []);

    // Fleet synchronization handler
    const handleFleetSync = useCallback(async (pointCloud: PointCloudData) => {
        try {
            const syncStart = performance.now();
            await fleetManager.syncState({ type: 'LIDAR_UPDATE', data: pointCloud });
            const syncLatency = performance.now() - syncStart;

            setPerformanceMetrics(prev => ({
                ...prev,
                fleetSyncLatency: syncLatency
            }));
        } catch (error) {
            console.error('Fleet sync error:', error);
        }
    }, [fleetManager]);

    // Scan update handler
    const handleScanUpdate = useCallback(async (scanData: PointCloudData) => {
        try {
            const processStart = performance.now();
            const processedData = await lidarService.processPointCloud(scanData.rawData);
            const processingLatency = performance.now() - processStart;

            setCurrentPointCloud(processedData);
            setPerformanceMetrics(prev => ({
                ...prev,
                processingLatency,
                pointsProcessed: processedData.points.length,
                quality: processingLatency <= 50 ? ScanQuality.HIGH : ScanQuality.MEDIUM
            }));

            // Sync with fleet if processing meets latency requirements
            if (processingLatency <= 50) {
                await handleFleetSync(processedData);
            }
        } catch (error) {
            console.error('Scan update error:', error);
        }
    }, [lidarService, handleFleetSync]);

    // Start scanning function
    const startScanning = useCallback(async () => {
        try {
            await lidarService.startScanning({
                frequency: 30,
                resolution: 0.01,
                range: 5.0
            });

            setScanState(prev => ({ ...prev, isActive: true }));
        } catch (error) {
            console.error('Failed to start scanning:', error);
            throw error;
        }
    }, [lidarService]);

    // Stop scanning function
    const stopScanning = useCallback(async () => {
        try {
            await lidarService.stopScanning();
            setScanState(prev => ({ ...prev, isActive: false }));
        } catch (error) {
            console.error('Failed to stop scanning:', error);
            throw error;
        }
    }, [lidarService]);

    // Update visualization configuration
    const updateVisualization = useCallback(async (config: any) => {
        try {
            await lidarService.updateVisualizationConfig(config);
        } catch (error) {
            console.error('Failed to update visualization:', error);
            throw error;
        }
    }, [lidarService]);

    // Setup event listeners and cleanup
    useEffect(() => {
        const scanSubscription = lidarService.subscribe('scanUpdate', handleScanUpdate);
        const fleetEventHandler = fleetManager.handleFleetEvent((event) => {
            setFleetState(prev => ({
                ...prev,
                membersCount: event.membersCount,
                syncStatus: event.status
            }));
        });

        // Performance monitoring interval
        const monitorInterval = setInterval(() => {
            const config = lidarService.getVisualizationConfig();
            updateVisualization(config).catch(console.error);
        }, PERFORMANCE_MONITOR_INTERVAL);

        // Cleanup function
        return () => {
            const cleanup = async () => {
                clearInterval(monitorInterval);
                if (scanState.isActive) {
                    await stopScanning();
                }
                lidarService.unsubscribe('scanUpdate', handleScanUpdate);
            };
            cleanup().catch(console.error);
        };
    }, [lidarService, fleetManager, handleScanUpdate, scanState.isActive, stopScanning]);

    return {
        scanState,
        fleetState,
        startScanning,
        stopScanning,
        updateVisualization,
        currentPointCloud,
        performanceMetrics
    };
}

export default useLidar;