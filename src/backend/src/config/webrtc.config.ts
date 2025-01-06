// WebRTC configuration for TALD UNIA platform - version: M98
import { RTCConfiguration } from 'webrtc'; // version: M98
import { MeshConfig } from '../types/mesh.types';
import { FleetSyncConfig } from '../types/fleet.types';

/**
 * Maximum timeout for ICE candidate gathering in milliseconds
 */
const MAX_ICE_GATHERING_TIMEOUT = 5000;

/**
 * Default ICE candidate pool size for connection optimization
 */
const DEFAULT_ICE_CANDIDATE_POOL_SIZE = 10;

/**
 * Primary STUN servers for NAT traversal
 */
const STUN_SERVERS = [
    'stun:stun1.l.google.com:19302',
    'stun:stun2.l.google.com:19302'
];

/**
 * TURN servers for fallback relay when direct P2P fails
 */
const TURN_SERVERS = [
    'turn:turn.tald-unia.com:3478'
];

/**
 * Maximum number of reconnection attempts
 */
const MAX_RECONNECT_ATTEMPTS = 3;

/**
 * Minimum acceptable connection quality score (0-1)
 */
const CONNECTION_QUALITY_THRESHOLD = 0.8;

/**
 * Interval for latency monitoring in milliseconds
 */
const LATENCY_MONITOR_INTERVAL = 1000;

/**
 * Core WebRTC configuration for P2P connections
 */
export const webrtcConfig: RTCConfiguration = {
    iceServers: [
        ...STUN_SERVERS.map(url => ({ urls: url })),
        {
            urls: TURN_SERVERS,
            username: process.env.TURN_USERNAME,
            credential: process.env.TURN_CREDENTIAL
        }
    ],
    bundlePolicy: 'max-bundle',
    iceCandidatePoolSize: DEFAULT_ICE_CANDIDATE_POOL_SIZE,
    iceTransportPolicy: 'all',
    rtcpMuxPolicy: 'require',
    // Enable DTLS for secure communication
    certificates: undefined, // Will be generated automatically
    iceServersTransportPolicy: 'all'
};

/**
 * Mesh network specific configuration
 */
export const meshNetworkConfig: MeshConfig = {
    maxPeers: 32, // Maximum fleet size requirement
    maxLatency: 50, // Maximum P2P latency requirement in ms
    iceServers: webrtcConfig.iceServers,
    reconnectTimeout: MAX_ICE_GATHERING_TIMEOUT,
    gatheringTimeout: MAX_ICE_GATHERING_TIMEOUT
};

/**
 * Fleet synchronization configuration
 */
export const fleetSyncConfig: FleetSyncConfig = {
    syncInterval: 50, // 50ms sync interval for real-time updates
    maxRetries: MAX_RECONNECT_ATTEMPTS,
    timeout: 1000, // 1 second timeout for sync operations
    retryBackoffStrategy: 'EXPONENTIAL',
    performanceThreshold: CONNECTION_QUALITY_THRESHOLD,
    networkQualityMetrics: true
};

/**
 * Generates optimal ICE server configuration based on network conditions and region
 * @param region Target deployment region
 * @param networkQuality Current network quality metrics
 * @returns Optimized ICE server configuration
 */
export function getOptimalIceServers(
    region: string[],
    networkQuality: { latency: number; stability: number }
): RTCIceServer[] {
    const baseServers = [...webrtcConfig.iceServers];
    
    // Add region-specific STUN servers if available
    if (region.includes('NA')) {
        baseServers.unshift({ urls: 'stun:stun-na.tald-unia.com:19302' });
    }
    if (region.includes('EU')) {
        baseServers.unshift({ urls: 'stun:stun-eu.tald-unia.com:19302' });
    }

    // Prioritize TURN servers if network quality is poor
    if (networkQuality.stability < CONNECTION_QUALITY_THRESHOLD) {
        const regionTurnServer = region.includes('EU') 
            ? 'turn:turn-eu.tald-unia.com:3478'
            : 'turn:turn-na.tald-unia.com:3478';
            
        baseServers.unshift({
            urls: regionTurnServer,
            username: process.env.TURN_USERNAME,
            credential: process.env.TURN_CREDENTIAL
        });
    }

    return baseServers;
}

/**
 * Validates WebRTC configuration against platform requirements
 * @param config WebRTC configuration to validate
 * @param metrics Current network metrics
 * @returns Validation result with detailed assessment
 */
export function validateWebRTCConfig(
    config: RTCConfiguration,
    metrics: { latency: number; bandwidth: number }
): { isValid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Validate ICE server configuration
    if (!config.iceServers || config.iceServers.length === 0) {
        issues.push('Missing ICE servers configuration');
    }

    // Validate latency requirements
    if (metrics.latency > meshNetworkConfig.maxLatency) {
        issues.push(`Latency ${metrics.latency}ms exceeds maximum threshold of ${meshNetworkConfig.maxLatency}ms`);
    }

    // Validate bundle policy
    if (config.bundlePolicy !== 'max-bundle') {
        issues.push('Bundle policy must be set to max-bundle for optimal performance');
    }

    // Validate ICE candidate pool size
    if (config.iceCandidatePoolSize !== DEFAULT_ICE_CANDIDATE_POOL_SIZE) {
        issues.push('Suboptimal ICE candidate pool size configuration');
    }

    return {
        isValid: issues.length === 0,
        issues
    };
}

/**
 * Monitors and optimizes WebRTC connection quality in real-time
 * @param connection Active WebRTC peer connection
 * @returns Real-time connection quality metrics
 */
export function monitorConnectionQuality(
    connection: RTCPeerConnection
): Promise<{ quality: number; latency: number }> {
    return new Promise((resolve) => {
        const stats = connection.getStats();
        
        stats.then(statsReport => {
            let totalRoundTripTime = 0;
            let sampleCount = 0;
            let packetsLost = 0;
            let packetsReceived = 0;

            statsReport.forEach(report => {
                if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                    totalRoundTripTime += report.currentRoundTripTime;
                    sampleCount++;
                }
                if (report.type === 'inbound-rtp') {
                    packetsLost += report.packetsLost;
                    packetsReceived += report.packetsReceived;
                }
            });

            const averageLatency = sampleCount > 0 ? totalRoundTripTime / sampleCount : 0;
            const packetLossRate = packetsReceived > 0 
                ? packetsLost / (packetsLost + packetsReceived)
                : 0;
            
            const quality = Math.max(0, 1 - (packetLossRate * 2) - (averageLatency / meshNetworkConfig.maxLatency));

            resolve({
                quality,
                latency: averageLatency * 1000 // Convert to milliseconds
            });
        });
    });
}