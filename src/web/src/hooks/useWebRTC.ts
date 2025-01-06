import { useState, useEffect, useCallback, useRef } from 'react'; // @version 18.2.0
import WebRTCService from '../services/webrtc.service';
import webrtcConfig from '../config/webrtc.config';
import {
    FleetMemberConnection,
    FleetNetworkStats,
    FleetSyncMessage,
    FleetStatus,
    FleetRole,
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD,
    isValidFleetSize,
    isValidLatency
} from '../types/fleet.types';

/**
 * React hook for managing WebRTC peer-to-peer connections with enhanced fleet management
 * and performance monitoring capabilities.
 */
export const useWebRTC = () => {
    // Service and connection state
    const webrtcService = useRef<WebRTCService>(new WebRTCService());
    const [isConnected, setIsConnected] = useState<boolean>(false);
    const [isFleetLeader, setIsFleetLeader] = useState<boolean>(false);
    const [fleetStatus, setFleetStatus] = useState<FleetStatus>(FleetStatus.INACTIVE);
    const [networkStats, setNetworkStats] = useState<FleetNetworkStats>({
        averageLatency: 0,
        maxLatency: 0,
        minLatency: Number.MAX_VALUE,
        packetsLost: 0,
        bandwidth: 0,
        connectedPeers: 0,
        syncLatency: 0
    });

    // Connection quality monitoring
    const [connectionQuality, setConnectionQuality] = useState<{
        score: number;
        status: 'excellent' | 'good' | 'fair' | 'poor';
    }>({ score: 1, status: 'excellent' });

    // Stats monitoring interval
    const statsIntervalRef = useRef<NodeJS.Timer>();

    /**
     * Initializes and connects to a fleet with the specified peer IDs
     */
    const connectToFleet = useCallback(async (peerIds: string[]) => {
        try {
            if (!isValidFleetSize(peerIds.length + 1)) {
                throw new Error(`Fleet size exceeds maximum of ${MAX_FLEET_SIZE} devices`);
            }

            await webrtcService.current.connectToFleet(peerIds);
            setIsConnected(true);
            setFleetStatus(FleetStatus.ACTIVE);

            // Start monitoring network stats
            statsIntervalRef.current = setInterval(async () => {
                const stats = await webrtcService.current.getNetworkStats();
                setNetworkStats(stats);
                updateConnectionQuality(stats);
            }, webrtcConfig.performance.adaptiveThresholds.samplingInterval);

        } catch (error) {
            console.error('Fleet connection error:', error);
            setIsConnected(false);
            setFleetStatus(FleetStatus.INACTIVE);
            throw error;
        }
    }, []);

    /**
     * Sends game state updates to connected fleet members
     */
    const sendGameState = useCallback(async (message: FleetSyncMessage) => {
        if (!isConnected) {
            throw new Error('Not connected to fleet');
        }

        try {
            await webrtcService.current.sendGameState(message);
            const stats = await webrtcService.current.getNetworkStats();
            setNetworkStats(stats);
        } catch (error) {
            console.error('Game state sync error:', error);
            throw error;
        }
    }, [isConnected]);

    /**
     * Updates connection quality metrics based on network statistics
     */
    const updateConnectionQuality = useCallback((stats: FleetNetworkStats) => {
        const latencyScore = Math.max(0, 1 - (stats.averageLatency / MAX_LATENCY_THRESHOLD));
        const packetLossScore = Math.max(0, 1 - (stats.packetsLost / 100));
        const bandwidthScore = Math.min(1, stats.bandwidth / webrtcConfig.performance.minBandwidth);

        const qualityScore = (latencyScore + packetLossScore + bandwidthScore) / 3;
        let status: 'excellent' | 'good' | 'fair' | 'poor' = 'poor';

        if (qualityScore >= 0.9) status = 'excellent';
        else if (qualityScore >= 0.7) status = 'good';
        else if (qualityScore >= 0.5) status = 'fair';

        setConnectionQuality({ score: qualityScore, status });
    }, []);

    /**
     * Handles fleet role changes and leadership status
     */
    const handleFleetRoleChange = useCallback((role: FleetRole) => {
        setIsFleetLeader(role === FleetRole.LEADER);
    }, []);

    /**
     * Cleanup function for WebRTC connections and intervals
     */
    const cleanup = useCallback(async () => {
        if (statsIntervalRef.current) {
            clearInterval(statsIntervalRef.current);
        }

        if (isConnected) {
            await webrtcService.current.disconnectFromFleet();
            setIsConnected(false);
            setFleetStatus(FleetStatus.INACTIVE);
            setIsFleetLeader(false);
        }
    }, [isConnected]);

    // Setup and cleanup effect
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, [cleanup]);

    return {
        connectToFleet,
        sendGameState,
        networkStats,
        isConnected,
        isFleetLeader,
        fleetStatus,
        connectionQuality
    };
};

export default useWebRTC;