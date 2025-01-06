import { RTCPeerConnection, RTCDataChannel, RTCSessionDescription } from 'webrtc'; // @version M98
import webrtcConfig from '../config/webrtc.config';
import { 
    FleetMemberConnection, 
    FleetNetworkStats, 
    FleetSyncMessage,
    FleetMessageType,
    MAX_LATENCY_THRESHOLD
} from '../types/fleet.types';

// Constants for WebRTC optimization
const ICE_GATHERING_TIMEOUT = 5000;
const MAX_LATENCY = 50;
const PING_INTERVAL = 1000;
const ROLLING_AVERAGE_WINDOW = 100;
const MAX_RETRY_ATTEMPTS = 3;
const BANDWIDTH_PER_PEER = 500000; // 500 Kbps per peer

/**
 * Creates and initializes a WebRTC peer connection with gaming optimizations
 * @param config RTCConfiguration for the peer connection
 * @param fleetMember Fleet member connection details
 * @returns Promise<RTCPeerConnection> Initialized peer connection
 */
export async function createPeerConnection(
    config: RTCConfiguration,
    fleetMember: FleetMemberConnection
): Promise<RTCPeerConnection> {
    const peerConnection = new RTCPeerConnection({
        ...webrtcConfig.peerConnection,
        ...config
    });

    // Configure connection monitoring
    let connectionStartTime: number;
    const latencyBuffer: number[] = [];

    peerConnection.addEventListener('connectionstatechange', () => {
        switch (peerConnection.connectionState) {
            case 'connecting':
                connectionStartTime = performance.now();
                break;
            case 'connected':
                const connectionTime = performance.now() - connectionStartTime;
                fleetMember.connectionQuality = calculateConnectionQuality(connectionTime);
                break;
            case 'failed':
                handleConnectionFailure(peerConnection, fleetMember);
                break;
        }
    });

    // Configure ICE candidate handling with regional optimization
    peerConnection.addEventListener('icecandidate', (event) => {
        if (event.candidate) {
            const candidateType = parseCandidateType(event.candidate.candidate);
            prioritizeCandidateByRegion(event.candidate, config);
        }
    });

    // Set up bandwidth allocation
    const transceivers = peerConnection.getTransceivers();
    transceivers.forEach(transceiver => {
        if (transceiver.sender) {
            const parameters = transceiver.sender.getParameters();
            if (parameters.encodings) {
                parameters.encodings[0].maxBitrate = BANDWIDTH_PER_PEER;
                transceiver.sender.setParameters(parameters);
            }
        }
    });

    return peerConnection;
}

/**
 * Creates an optimized WebRTC data channel for game state synchronization
 * @param peerConnection RTCPeerConnection instance
 * @param label Channel label/identifier
 * @param options DataChannel configuration options
 * @returns Promise<RTCDataChannel> Configured data channel
 */
export async function createDataChannel(
    peerConnection: RTCPeerConnection,
    label: string,
    options: RTCDataChannelInit = webrtcConfig.dataChannel.game
): Promise<RTCDataChannel> {
    const dataChannel = peerConnection.createDataChannel(label, {
        ordered: true,
        maxRetransmits: options.maxRetransmits,
        priority: 'high'
    });

    // Configure message handling with priority
    dataChannel.addEventListener('message', async (event) => {
        const message: FleetSyncMessage = JSON.parse(event.data);
        await handlePrioritizedMessage(message, dataChannel);
    });

    // Set up error recovery
    dataChannel.addEventListener('error', (error) => {
        handleDataChannelError(error, dataChannel, peerConnection);
    });

    // Configure automatic bandwidth adaptation
    setupBandwidthMonitoring(dataChannel);

    return dataChannel;
}

/**
 * Measures peer-to-peer latency with advanced statistical analysis
 * @param dataChannel RTCDataChannel instance
 * @param networkStats Current network statistics
 * @returns Promise<number> Measured latency in milliseconds
 */
export async function measureLatency(
    dataChannel: RTCDataChannel,
    networkStats: FleetNetworkStats
): Promise<number> {
    const measurements: number[] = [];
    let pingCount = 0;

    return new Promise((resolve) => {
        const pingInterval = setInterval(() => {
            if (pingCount >= ROLLING_AVERAGE_WINDOW) {
                clearInterval(pingInterval);
                const latency = calculateOptimizedLatency(measurements, networkStats);
                resolve(latency);
                return;
            }

            const pingStart = performance.now();
            const pingMessage: FleetSyncMessage = {
                type: FleetMessageType.PING,
                payload: { timestamp: pingStart },
                timestamp: Date.now(),
                senderId: dataChannel.label,
                sequence: pingCount,
                priority: 9
            };

            dataChannel.send(JSON.stringify(pingMessage));
            
            dataChannel.addEventListener('message', function onPong(event) {
                const pongData = JSON.parse(event.data);
                if (pongData.type === FleetMessageType.PING) {
                    const latency = performance.now() - pongData.payload.timestamp;
                    measurements.push(latency);
                    pingCount++;
                    dataChannel.removeEventListener('message', onPong);
                }
            });
        }, PING_INTERVAL);
    });
}

// Helper functions

function calculateConnectionQuality(connectionTime: number): number {
    const maxAcceptableTime = 1000; // 1 second
    return Math.max(0, 1 - (connectionTime / maxAcceptableTime));
}

async function handleConnectionFailure(
    peerConnection: RTCPeerConnection,
    fleetMember: FleetMemberConnection
): Promise<void> {
    if (fleetMember.retryCount < MAX_RETRY_ATTEMPTS) {
        fleetMember.retryCount++;
        await peerConnection.restartIce();
    }
}

function parseCandidateType(candidateStr: string): string {
    const match = candidateStr.match(/typ ([a-z]+)/);
    return match ? match[1] : '';
}

function prioritizeCandidateByRegion(
    candidate: RTCIceCandidate,
    config: RTCConfiguration
): void {
    const candidateIP = candidate.address || candidate.candidate.split(' ')[4];
    // Implement regional prioritization logic
}

async function handlePrioritizedMessage(
    message: FleetSyncMessage,
    dataChannel: RTCDataChannel
): Promise<void> {
    if (message.priority >= 8) { // High priority messages
        await processHighPriorityMessage(message);
    } else {
        queueMessageProcessing(message);
    }
}

function handleDataChannelError(
    error: RTCErrorEvent,
    dataChannel: RTCDataChannel,
    peerConnection: RTCPeerConnection
): void {
    console.error(`DataChannel Error: ${error.error.message}`);
    if (dataChannel.readyState === 'closed') {
        attemptChannelRecovery(dataChannel, peerConnection);
    }
}

function setupBandwidthMonitoring(dataChannel: RTCDataChannel): void {
    let bytesReceived = 0;
    let lastCheckTime = performance.now();

    dataChannel.addEventListener('message', (event) => {
        bytesReceived += event.data.length;
        const now = performance.now();
        if (now - lastCheckTime >= 1000) {
            const bandwidth = (bytesReceived * 8) / ((now - lastCheckTime) / 1000);
            adaptChannelParameters(dataChannel, bandwidth);
            bytesReceived = 0;
            lastCheckTime = now;
        }
    });
}

function calculateOptimizedLatency(
    measurements: number[],
    networkStats: FleetNetworkStats
): number {
    // Remove outliers
    const sortedMeasurements = [...measurements].sort((a, b) => a - b);
    const q1 = sortedMeasurements[Math.floor(measurements.length * 0.25)];
    const q3 = sortedMeasurements[Math.floor(measurements.length * 0.75)];
    const iqr = q3 - q1;
    const validMeasurements = measurements.filter(
        m => m >= q1 - 1.5 * iqr && m <= q3 + 1.5 * iqr
    );

    // Calculate weighted average
    const latency = validMeasurements.reduce((sum, val) => sum + val, 0) / validMeasurements.length;
    return Math.min(latency, MAX_LATENCY_THRESHOLD);
}

async function processHighPriorityMessage(message: FleetSyncMessage): Promise<void> {
    // Implement high-priority message processing
}

function queueMessageProcessing(message: FleetSyncMessage): void {
    // Implement message queuing for lower priority messages
}

async function attemptChannelRecovery(
    dataChannel: RTCDataChannel,
    peerConnection: RTCPeerConnection
): Promise<void> {
    // Implement channel recovery logic
}

function adaptChannelParameters(
    dataChannel: RTCDataChannel,
    bandwidth: number
): void {
    // Implement adaptive channel parameters based on bandwidth
}