import React, { useState, useEffect, useCallback, useMemo } from 'react'; // @version 18.0.0
import { 
    Card, 
    Slider, 
    Switch, 
    Button, 
    Typography, 
    Grid, 
    CircularProgress, 
    Alert 
} from '@mui/material'; // @version ^5.14.0

import { useWebRTC } from '../../hooks/useWebRTC';
import { 
    PERFORMANCE_THRESHOLDS, 
    DATA_CHANNEL_CONFIG, 
    regionalSettings 
} from '../../config/webrtc.config';
import { 
    IFleet, 
    FleetNetworkStats, 
    NetworkRegion 
} from '../../interfaces/fleet.interface';

interface NetworkSettingsProps {
    fleetId?: string;
    onSettingsChange: (settings: NetworkSettingsState) => void;
    region: NetworkRegion;
    onRegionChange: (region: NetworkRegion) => void;
}

interface NetworkSettingsState {
    autoJoinFleet: boolean;
    maxLatency: number;
    maxPacketLoss: number;
    minBandwidth: number;
    enableQos: boolean;
    selectedRegion: NetworkRegion;
    optimizationLevel: OptimizationLevel;
    fleetSize: number;
}

type OptimizationLevel = 'AUTO' | 'PERFORMANCE' | 'BATTERY' | 'BALANCED';

const DEFAULT_SETTINGS: NetworkSettingsState = {
    autoJoinFleet: true,
    maxLatency: PERFORMANCE_THRESHOLDS.maxLatency,
    maxPacketLoss: PERFORMANCE_THRESHOLDS.maxPacketLoss,
    minBandwidth: PERFORMANCE_THRESHOLDS.minBandwidth,
    enableQos: true,
    selectedRegion: 'na',
    optimizationLevel: 'AUTO',
    fleetSize: 32
};

const NetworkSettings: React.FC<NetworkSettingsProps> = ({
    fleetId,
    onSettingsChange,
    region,
    onRegionChange
}) => {
    // State management
    const [settings, setSettings] = useState<NetworkSettingsState>(DEFAULT_SETTINGS);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // WebRTC hook for network monitoring
    const { networkStats, connectionQuality, isConnected } = useWebRTC();

    // Memoized regional thresholds
    const regionalThresholds = useMemo(() => 
        PERFORMANCE_THRESHOLDS.regional[settings.selectedRegion], 
        [settings.selectedRegion]
    );

    // Initialize settings with regional defaults
    useEffect(() => {
        setSettings(prev => ({
            ...prev,
            maxLatency: regionalThresholds.maxLatency,
            maxPacketLoss: regionalThresholds.maxPacketLoss,
            selectedRegion: region
        }));
    }, [region, regionalThresholds]);

    // Handle latency changes with validation
    const handleLatencyChange = useCallback((value: number) => {
        if (value <= PERFORMANCE_THRESHOLDS.maxLatency) {
            setSettings(prev => ({
                ...prev,
                maxLatency: value
            }));
            onSettingsChange({
                ...settings,
                maxLatency: value
            });
        }
    }, [settings, onSettingsChange]);

    // Handle region changes with optimization
    const handleRegionChange = useCallback((newRegion: NetworkRegion) => {
        setSettings(prev => ({
            ...prev,
            selectedRegion: newRegion,
            maxLatency: PERFORMANCE_THRESHOLDS.regional[newRegion].maxLatency,
            maxPacketLoss: PERFORMANCE_THRESHOLDS.regional[newRegion].maxPacketLoss
        }));
        onRegionChange(newRegion);
    }, [onRegionChange]);

    // Handle QoS toggle
    const handleQosToggle = useCallback((checked: boolean) => {
        setSettings(prev => ({
            ...prev,
            enableQos: checked
        }));
        onSettingsChange({
            ...settings,
            enableQos: checked
        });
    }, [settings, onSettingsChange]);

    // Handle optimization level change
    const handleOptimizationChange = useCallback((level: OptimizationLevel) => {
        setSettings(prev => ({
            ...prev,
            optimizationLevel: level
        }));
        onSettingsChange({
            ...settings,
            optimizationLevel: level
        });
    }, [settings, onSettingsChange]);

    // Network quality indicator
    const getQualityColor = useMemo(() => {
        if (connectionQuality.score >= 0.9) return 'success';
        if (connectionQuality.score >= 0.7) return 'info';
        if (connectionQuality.score >= 0.5) return 'warning';
        return 'error';
    }, [connectionQuality.score]);

    return (
        <Card sx={{ p: 3 }}>
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={3}>
                {/* Region Selection */}
                <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                        Region Settings
                    </Typography>
                    <Grid container spacing={2}>
                        {Object.keys(PERFORMANCE_THRESHOLDS.regional).map((regionKey) => (
                            <Grid item key={regionKey}>
                                <Button
                                    variant={settings.selectedRegion === regionKey ? 'contained' : 'outlined'}
                                    onClick={() => handleRegionChange(regionKey as NetworkRegion)}
                                >
                                    {regionKey.toUpperCase()}
                                </Button>
                            </Grid>
                        ))}
                    </Grid>
                </Grid>

                {/* Performance Settings */}
                <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                        Performance Settings
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Typography>
                                Maximum Latency ({settings.maxLatency}ms)
                            </Typography>
                            <Slider
                                value={settings.maxLatency}
                                onChange={(_, value) => handleLatencyChange(value as number)}
                                min={20}
                                max={PERFORMANCE_THRESHOLDS.maxLatency}
                                step={1}
                                marks
                                disabled={isLoading}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography>
                                Packet Loss Threshold ({(settings.maxPacketLoss * 100).toFixed(1)}%)
                            </Typography>
                            <Slider
                                value={settings.maxPacketLoss}
                                onChange={(_, value) => setSettings(prev => ({ ...prev, maxPacketLoss: value as number }))}
                                min={0}
                                max={0.2}
                                step={0.01}
                                marks
                                disabled={isLoading}
                            />
                        </Grid>
                    </Grid>
                </Grid>

                {/* Fleet Settings */}
                <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                        Fleet Configuration
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <Switch
                                checked={settings.autoJoinFleet}
                                onChange={(e) => setSettings(prev => ({ ...prev, autoJoinFleet: e.target.checked }))}
                                disabled={isLoading}
                            />
                            <Typography component="span" sx={{ ml: 1 }}>
                                Auto-join Fleet
                            </Typography>
                        </Grid>
                        <Grid item xs={12}>
                            <Typography>
                                Fleet Size Limit ({settings.fleetSize} devices)
                            </Typography>
                            <Slider
                                value={settings.fleetSize}
                                onChange={(_, value) => setSettings(prev => ({ ...prev, fleetSize: value as number }))}
                                min={2}
                                max={32}
                                step={1}
                                marks
                                disabled={isLoading}
                            />
                        </Grid>
                    </Grid>
                </Grid>

                {/* Network Quality */}
                <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                        Network Quality
                    </Typography>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item>
                            <CircularProgress
                                variant="determinate"
                                value={connectionQuality.score * 100}
                                color={getQualityColor}
                                size={40}
                            />
                        </Grid>
                        <Grid item>
                            <Typography variant="body1">
                                {connectionQuality.status.toUpperCase()} ({(connectionQuality.score * 100).toFixed(1)}%)
                            </Typography>
                        </Grid>
                    </Grid>
                </Grid>

                {/* Real-time Stats */}
                {isConnected && (
                    <Grid item xs={12}>
                        <Typography variant="h6" gutterBottom>
                            Real-time Statistics
                        </Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={6}>
                                <Typography>Current Latency: {networkStats.averageLatency.toFixed(1)}ms</Typography>
                            </Grid>
                            <Grid item xs={6}>
                                <Typography>Packet Loss: {(networkStats.packetsLost * 100).toFixed(2)}%</Typography>
                            </Grid>
                            <Grid item xs={6}>
                                <Typography>Bandwidth: {(networkStats.bandwidth / 1000000).toFixed(2)} Mbps</Typography>
                            </Grid>
                            <Grid item xs={6}>
                                <Typography>Connected Peers: {networkStats.connectedPeers}</Typography>
                            </Grid>
                        </Grid>
                    </Grid>
                )}
            </Grid>
        </Card>
    );
};

export default NetworkSettings;