import { RTCPeerConnection } from 'webrtc'; // @version M98

// Maximum number of concurrent peers in a fleet
const MAX_PEERS = 32;

// Timeout configurations (in milliseconds)
const RECONNECT_TIMEOUT = 5000;
const ICE_GATHERING_TIMEOUT = 5000;
const TURN_CREDENTIAL_ROTATION_INTERVAL = 86400000;

// ICE Server configurations
const ICE_SERVERS: RTCIceServer[] = [
    {
        urls: [
            'stun:stun1.l.google.com:19302',
            'stun:stun2.l.google.com:19302',
            'stun:stun3.l.google.com:19302',
            'stun:stun4.l.google.com:19302'
        ]
    },
    {
        urls: [
            'turn:na.turn.tald-unia.com:3478',
            'turn:eu.turn.tald-unia.com:3478',
            'turn:ap.turn.tald-unia.com:3478'
        ],
        username: process.env.TURN_USERNAME,
        credential: process.env.TURN_CREDENTIAL,
        credentialType: 'password'
    }
];

// Signaling server configuration
const SIGNALING_CONFIG = {
    urls: {
        primary: 'wss://signaling.tald-unia.com',
        fallback: 'wss://fallback.signaling.tald-unia.com'
    },
    reconnectInterval: 1000,
    maxReconnectAttempts: 5,
    heartbeatInterval: 5000,
    connectionTimeout: 10000
};

// Peer connection configuration
const PEER_CONNECTION_CONFIG: RTCConfiguration = {
    iceServers: ICE_SERVERS,
    iceTransportPolicy: 'all',
    bundlePolicy: 'max-bundle',
    rtcpMuxPolicy: 'require',
    iceCandidatePoolSize: 10,
    sdpSemantics: 'unified-plan',
    encodedInsertableStreams: true
};

// Data channel configurations for different purposes
const DATA_CHANNEL_CONFIG = {
    ordered: true,
    maxRetransmits: 0,
    maxPacketLifeTime: 1000,
    priority: 'high',
    channels: {
        game: {
            ordered: true,
            maxRetransmits: 0 // No retransmission for real-time game state
        },
        crdt: {
            ordered: true,
            maxRetransmits: 3 // Limited retransmission for CRDT sync
        },
        chat: {
            ordered: true,
            maxRetransmits: 5 // More retransmissions for chat reliability
        }
    }
};

// Performance thresholds by region
const PERFORMANCE_THRESHOLDS = {
    maxLatency: 50, // Global max latency in ms
    maxPacketLoss: 0.1, // 10% maximum packet loss
    minBandwidth: 1_000_000, // Minimum 1 Mbps bandwidth
    regional: {
        na: {
            maxLatency: 45,
            maxPacketLoss: 0.08
        },
        eu: {
            maxLatency: 48,
            maxPacketLoss: 0.09
        },
        ap: {
            maxLatency: 50,
            maxPacketLoss: 0.1
        }
    },
    adaptiveThresholds: {
        enabled: true,
        samplingInterval: 1000,
        adjustmentFactor: 1.2
    }
};

interface PerformanceMetrics {
    latency: number;
    packetLoss: number;
    bandwidth: number;
    region: 'na' | 'eu' | 'ap';
}

const createDefaultConfig = (region: string, networkType: string): RTCConfiguration => {
    const config = { ...PEER_CONNECTION_CONFIG };
    
    // Select regional ICE servers
    config.iceServers = ICE_SERVERS.map(server => ({
        ...server,
        urls: Array.isArray(server.urls) 
            ? server.urls.filter(url => url.includes(region))
            : server.urls
    }));

    // Apply network-specific optimizations
    if (networkType === 'wifi') {
        config.iceCandidatePoolSize = 10;
    } else if (networkType === 'ethernet') {
        config.iceCandidatePoolSize = 5;
    }

    return config;
};

const updatePerformanceThresholds = (metrics: PerformanceMetrics): void => {
    const regional = PERFORMANCE_THRESHOLDS.regional[metrics.region];
    
    if (PERFORMANCE_THRESHOLDS.adaptiveThresholds.enabled) {
        const adjustmentFactor = PERFORMANCE_THRESHOLDS.adaptiveThresholds.adjustmentFactor;
        
        regional.maxLatency = Math.min(
            regional.maxLatency * adjustmentFactor,
            PERFORMANCE_THRESHOLDS.maxLatency
        );
        
        regional.maxPacketLoss = Math.min(
            regional.maxPacketLoss * adjustmentFactor,
            PERFORMANCE_THRESHOLDS.maxPacketLoss
        );
    }
};

export const webrtcConfig = {
    iceServers: ICE_SERVERS,
    signaling: SIGNALING_CONFIG,
    peerConnection: PEER_CONNECTION_CONFIG,
    dataChannel: DATA_CHANNEL_CONFIG,
    performance: PERFORMANCE_THRESHOLDS,
    createDefaultConfig,
    updatePerformanceThresholds,
    MAX_PEERS,
    RECONNECT_TIMEOUT,
    ICE_GATHERING_TIMEOUT,
    TURN_CREDENTIAL_ROTATION_INTERVAL
};

export default webrtcConfig;