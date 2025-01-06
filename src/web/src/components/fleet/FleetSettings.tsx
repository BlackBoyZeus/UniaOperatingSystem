import React, { useCallback, useMemo } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'react-toastify';
import { ErrorBoundary } from 'react-error-boundary';

// Internal imports
import { IFleet } from '../../interfaces/fleet.interface';
import { useFleet } from '../../hooks/useFleet';

// Constants
const MAX_FLEET_SIZE = 32;
const MIN_FLEET_SIZE = 2;
const NETWORK_QUALITY_THRESHOLD = {
    minLatency: 50,
    minBandwidth: 1000
};

// Form data interface
interface FleetSettingsFormData {
    maxDevices: number;
    isPublic: boolean;
    allowJoinRequests: boolean;
    enableVoiceChat: boolean;
    networkQualityThreshold: {
        latency: number;
        bandwidth: number;
    };
}

// Error fallback component
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
    <div role="alert" className="fleet-settings-error">
        <h2>Fleet Settings Error</h2>
        <pre>{error.message}</pre>
    </div>
);

/**
 * Enhanced Fleet Settings component with real-time monitoring and accessibility
 * @returns React.FC FleetSettings component
 */
const FleetSettings: React.FC = React.memo(() => {
    // Custom hook for fleet management
    const { currentFleet, networkStats, updateFleetSettings } = useFleet();

    // Form initialization with validation
    const { register, handleSubmit, formState: { errors }, setValue, watch } = useForm<FleetSettingsFormData>({
        defaultValues: useMemo(() => ({
            maxDevices: currentFleet?.maxDevices || MAX_FLEET_SIZE,
            isPublic: currentFleet?.isPublic || false,
            allowJoinRequests: currentFleet?.allowJoinRequests || true,
            enableVoiceChat: currentFleet?.enableVoiceChat || false,
            networkQualityThreshold: {
                latency: NETWORK_QUALITY_THRESHOLD.minLatency,
                bandwidth: NETWORK_QUALITY_THRESHOLD.minBandwidth
            }
        }), [currentFleet])
    });

    // Network quality indicator calculation
    const networkQuality = useMemo(() => {
        if (!networkStats) return 'unknown';
        const qualityScore = (networkStats.averageLatency <= NETWORK_QUALITY_THRESHOLD.minLatency &&
            networkStats.bandwidth >= NETWORK_QUALITY_THRESHOLD.minBandwidth);
        return qualityScore ? 'optimal' : 'degraded';
    }, [networkStats]);

    // Enhanced form submission handler with validation and error handling
    const onSubmit = useCallback(async (data: FleetSettingsFormData) => {
        try {
            // Validate fleet size constraints
            if (data.maxDevices < MIN_FLEET_SIZE || data.maxDevices > MAX_FLEET_SIZE) {
                throw new Error(`Fleet size must be between ${MIN_FLEET_SIZE} and ${MAX_FLEET_SIZE}`);
            }

            // Validate network quality requirements
            if (networkStats && networkStats.averageLatency > data.networkQualityThreshold.latency) {
                toast.warning('Network latency exceeds recommended threshold', {
                    toastId: 'network-warning'
                });
            }

            // Update fleet settings
            await updateFleetSettings({
                ...data,
                networkQualityThreshold: {
                    latency: Math.min(data.networkQualityThreshold.latency, NETWORK_QUALITY_THRESHOLD.minLatency),
                    bandwidth: Math.max(data.networkQualityThreshold.bandwidth, NETWORK_QUALITY_THRESHOLD.minBandwidth)
                }
            });

            toast.success('Fleet settings updated successfully', {
                toastId: 'settings-success'
            });

        } catch (error) {
            console.error('Failed to update fleet settings:', error);
            toast.error('Failed to update fleet settings', {
                toastId: 'settings-error'
            });
        }
    }, [networkStats, updateFleetSettings]);

    if (!currentFleet) {
        return <div role="alert">No active fleet found</div>;
    }

    return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
            <div className="fleet-settings" role="region" aria-label="Fleet Settings">
                <h2>Fleet Settings</h2>
                
                {/* Network Status Indicator */}
                <div className="network-status" role="status" aria-live="polite">
                    <h3>Network Quality: {networkQuality}</h3>
                    <div className="network-metrics">
                        <span>Latency: {networkStats?.averageLatency}ms</span>
                        <span>Bandwidth: {networkStats?.bandwidth}Mbps</span>
                        <span>Connected Peers: {networkStats?.connectedPeers}</span>
                    </div>
                </div>

                {/* Settings Form */}
                <form onSubmit={handleSubmit(onSubmit)} className="settings-form">
                    {/* Fleet Size Configuration */}
                    <div className="form-group">
                        <label htmlFor="maxDevices">
                            Maximum Devices (2-32)
                            <input
                                type="number"
                                id="maxDevices"
                                min={MIN_FLEET_SIZE}
                                max={MAX_FLEET_SIZE}
                                {...register('maxDevices', {
                                    required: true,
                                    min: MIN_FLEET_SIZE,
                                    max: MAX_FLEET_SIZE
                                })}
                                aria-invalid={errors.maxDevices ? 'true' : 'false'}
                            />
                        </label>
                        {errors.maxDevices && (
                            <span role="alert" className="error-message">
                                Invalid fleet size
                            </span>
                        )}
                    </div>

                    {/* Privacy Settings */}
                    <div className="form-group">
                        <label htmlFor="isPublic">
                            <input
                                type="checkbox"
                                id="isPublic"
                                {...register('isPublic')}
                            />
                            Public Fleet
                        </label>
                    </div>

                    {/* Join Request Settings */}
                    <div className="form-group">
                        <label htmlFor="allowJoinRequests">
                            <input
                                type="checkbox"
                                id="allowJoinRequests"
                                {...register('allowJoinRequests')}
                            />
                            Allow Join Requests
                        </label>
                    </div>

                    {/* Voice Chat Settings */}
                    <div className="form-group">
                        <label htmlFor="enableVoiceChat">
                            <input
                                type="checkbox"
                                id="enableVoiceChat"
                                {...register('enableVoiceChat')}
                            />
                            Enable Voice Chat
                        </label>
                    </div>

                    {/* Network Quality Thresholds */}
                    <div className="form-group">
                        <h3>Network Quality Thresholds</h3>
                        <label htmlFor="latencyThreshold">
                            Maximum Latency (ms)
                            <input
                                type="number"
                                id="latencyThreshold"
                                {...register('networkQualityThreshold.latency')}
                                min={0}
                                max={100}
                            />
                        </label>
                        <label htmlFor="bandwidthThreshold">
                            Minimum Bandwidth (Mbps)
                            <input
                                type="number"
                                id="bandwidthThreshold"
                                {...register('networkQualityThreshold.bandwidth')}
                                min={100}
                            />
                        </label>
                    </div>

                    <button 
                        type="submit"
                        className="submit-button"
                        disabled={!networkStats || networkQuality === 'degraded'}
                    >
                        Update Settings
                    </button>
                </form>
            </div>
        </ErrorBoundary>
    );
});

FleetSettings.displayName = 'FleetSettings';

export default FleetSettings;