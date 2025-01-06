import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useHDRContext } from '@tald/hdr-context'; // ^1.0.0
import { usePerformanceMonitor } from '@tald/performance-monitor'; // ^1.0.0

import { ILidarScanConfig, ProcessingMode } from '../../interfaces/lidar.interface';
import { useLidar } from '../../hooks/useLidar';
import { LIDAR_SCAN_SETTINGS } from '../../constants/lidar.constants';

interface LidarSettingsProps {
    className?: string;
    onSettingsChange?: (config: ILidarScanConfig) => void;
    fleetConfig?: {
        syncEnabled: boolean;
        role: 'LEADER' | 'MEMBER';
    };
    powerPreferences?: {
        mode: ProcessingMode;
        adaptiveQuality: boolean;
    };
    hdrCapabilities?: {
        supported: boolean;
        maxNits: number;
    };
}

/**
 * Enhanced LiDAR settings component with power optimization and fleet support
 * Provides controls for scan frequency, resolution, range, and visualization
 * while enforcing system constraints and supporting fleet synchronization.
 */
export const LidarSettings: React.FC<LidarSettingsProps> = ({
    className,
    onSettingsChange,
    fleetConfig,
    powerPreferences,
    hdrCapabilities
}) => {
    // Custom hooks
    const { scanState, visualConfig, updateVisualizationConfig, fleetSync } = useLidar();
    const { isHDREnabled, brightness } = useHDRContext();
    const { addMetric, getMetrics } = usePerformanceMonitor();

    // Local state for settings
    const [scanConfig, setScanConfig] = useState<ILidarScanConfig>({
        scanFrequency: LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY,
        resolution: LIDAR_SCAN_SETTINGS.RESOLUTION,
        range: LIDAR_SCAN_SETTINGS.MAX_RANGE,
        adaptiveQuality: true,
        processingMode: ProcessingMode.REAL_TIME
    });

    // Performance metrics state
    const [performanceMetrics, setPerformanceMetrics] = useState({
        processingLatency: 0,
        qualityScore: 1,
        powerEfficiency: 1
    });

    // Memoized constraints based on power mode
    const constraints = useMemo(() => ({
        frequency: {
            min: 1,
            max: scanConfig.processingMode === ProcessingMode.POWER_SAVE ? 15 : 30
        },
        resolution: {
            min: LIDAR_SCAN_SETTINGS.RESOLUTION,
            max: scanConfig.processingMode === ProcessingMode.QUALITY ? 0.005 : 0.01
        },
        range: {
            min: 0.1,
            max: LIDAR_SCAN_SETTINGS.MAX_RANGE
        }
    }), [scanConfig.processingMode]);

    /**
     * Handles changes to scan frequency with power optimization
     */
    const handleFrequencyChange = useCallback((value: number) => {
        const start = performance.now();
        
        // Validate frequency within constraints
        const validFrequency = Math.min(
            Math.max(value, constraints.frequency.min),
            constraints.frequency.max
        );

        // Calculate power impact
        const powerImpact = validFrequency / constraints.frequency.max;
        
        setScanConfig(prev => ({
            ...prev,
            scanFrequency: validFrequency
        }));

        // Update performance metrics
        const latency = performance.now() - start;
        addMetric('frequencyUpdateLatency', latency);

        // Sync with fleet if enabled
        if (fleetConfig?.syncEnabled) {
            fleetSync.broadcastUpdate({
                type: 'SCAN_FREQUENCY_UPDATE',
                value: validFrequency,
                timestamp: Date.now()
            });
        }

        onSettingsChange?.({
            ...scanConfig,
            scanFrequency: validFrequency
        });
    }, [constraints, fleetConfig, fleetSync, onSettingsChange, scanConfig, addMetric]);

    /**
     * Handles changes to processing mode with quality adjustments
     */
    const handleProcessingModeChange = useCallback((mode: ProcessingMode) => {
        setScanConfig(prev => ({
            ...prev,
            processingMode: mode,
            // Adjust frequency based on mode
            scanFrequency: mode === ProcessingMode.POWER_SAVE 
                ? Math.min(prev.scanFrequency, 15)
                : prev.scanFrequency,
            // Adjust resolution based on mode
            resolution: mode === ProcessingMode.QUALITY
                ? Math.min(prev.resolution, 0.005)
                : prev.resolution
        }));

        // Update HDR settings if supported
        if (hdrCapabilities?.supported) {
            updateVisualizationConfig({
                ...visualConfig,
                brightness: mode === ProcessingMode.QUALITY ? brightness * 1.2 : brightness
            });
        }

        onSettingsChange?.(scanConfig);
    }, [scanConfig, hdrCapabilities, brightness, visualConfig, updateVisualizationConfig, onSettingsChange]);

    /**
     * Updates performance metrics and adjusts settings if needed
     */
    useEffect(() => {
        const updateMetrics = () => {
            const metrics = getMetrics();
            const newMetrics = {
                processingLatency: metrics.scanProcessing?.average || 0,
                qualityScore: metrics.scanQuality?.latest || 1,
                powerEfficiency: metrics.powerUsage?.average || 1
            };

            setPerformanceMetrics(newMetrics);

            // Auto-adjust settings based on metrics
            if (scanConfig.adaptiveQuality && newMetrics.processingLatency > 45) {
                handleProcessingModeChange(ProcessingMode.POWER_SAVE);
            }
        };

        const metricsInterval = setInterval(updateMetrics, 1000);
        return () => clearInterval(metricsInterval);
    }, [getMetrics, scanConfig.adaptiveQuality, handleProcessingModeChange]);

    return (
        <div className={className}>
            <div className="settings-group">
                <h3>Scan Frequency</h3>
                <input
                    type="range"
                    min={constraints.frequency.min}
                    max={constraints.frequency.max}
                    step={1}
                    value={scanConfig.scanFrequency}
                    onChange={(e) => handleFrequencyChange(Number(e.target.value))}
                    disabled={!scanState.isActive}
                />
                <span>{scanConfig.scanFrequency} Hz</span>
            </div>

            <div className="settings-group">
                <h3>Processing Mode</h3>
                <select
                    value={scanConfig.processingMode}
                    onChange={(e) => handleProcessingModeChange(e.target.value as ProcessingMode)}
                    disabled={!scanState.isActive}
                >
                    <option value={ProcessingMode.REAL_TIME}>Real-time</option>
                    <option value={ProcessingMode.QUALITY}>Quality</option>
                    <option value={ProcessingMode.POWER_SAVE}>Power Save</option>
                </select>
            </div>

            <div className="settings-group">
                <h3>Resolution</h3>
                <input
                    type="range"
                    min={constraints.resolution.min}
                    max={constraints.resolution.max}
                    step={0.001}
                    value={scanConfig.resolution}
                    onChange={(e) => {
                        const value = Number(e.target.value);
                        setScanConfig(prev => ({
                            ...prev,
                            resolution: value
                        }));
                        onSettingsChange?.({
                            ...scanConfig,
                            resolution: value
                        });
                    }}
                    disabled={!scanState.isActive}
                />
                <span>{scanConfig.resolution.toFixed(3)} cm</span>
            </div>

            <div className="settings-group">
                <h3>Scan Range</h3>
                <input
                    type="range"
                    min={constraints.range.min}
                    max={constraints.range.max}
                    step={0.1}
                    value={scanConfig.range}
                    onChange={(e) => {
                        const value = Number(e.target.value);
                        setScanConfig(prev => ({
                            ...prev,
                            range: value
                        }));
                        onSettingsChange?.({
                            ...scanConfig,
                            range: value
                        });
                    }}
                    disabled={!scanState.isActive}
                />
                <span>{scanConfig.range.toFixed(1)} m</span>
            </div>

            <div className="settings-group">
                <h3>Adaptive Quality</h3>
                <input
                    type="checkbox"
                    checked={scanConfig.adaptiveQuality}
                    onChange={(e) => {
                        setScanConfig(prev => ({
                            ...prev,
                            adaptiveQuality: e.target.checked
                        }));
                        onSettingsChange?.({
                            ...scanConfig,
                            adaptiveQuality: e.target.checked
                        });
                    }}
                    disabled={!scanState.isActive}
                />
            </div>

            <div className="metrics-display">
                <div>Processing Latency: {performanceMetrics.processingLatency.toFixed(1)} ms</div>
                <div>Quality Score: {(performanceMetrics.qualityScore * 100).toFixed(1)}%</div>
                <div>Power Efficiency: {(performanceMetrics.powerEfficiency * 100).toFixed(1)}%</div>
            </div>
        </div>
    );
};

export default LidarSettings;