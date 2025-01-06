import { RTCPeerConnection, RTCDataChannel, RTCSessionDescription } from 'webrtc'; // @version M98
import webrtcConfig from '../config/webrtc.config';
import { createPeerConnection, createDataChannel, measureLatency } from '../utils/webrtc.utils';
import {
    FleetMemberConnection,
    FleetNetworkStats,
    FleetSyncMessage,
    FleetMessageType,
    FleetStatus,
    FleetRole,
    MAX_FLEET_SIZE,
    DEFAULT_SYNC_INTERVAL,
    MAX_SYNC_RETRIES,
    SYNC_TIMEOUT,
    MIN_LEADER_SCORE
} from '../types/fleet.types';

// Service constants
const MAX_PEERS = 32;
const RECONNECT_INTERVAL = 1000;
const MAX_RECONNECT_ATTEMPTS = 5;
const NETWORK_STATS_INTERVAL = 1000;
const LEADER_ELECTION_TIMEOUT = 5000;
const BANDWIDTH_CHECK_INTERVAL = 2000;

export class WebRTCService {
    private peerConnections: Map<string, FleetMemberConnection>;
    private networkStats: FleetNetworkStats;
    private signalingConnection: WebSocket;
    private fleetLeaderId: string | null;
    private backupLeaderId: string | null;
    private peerLatencies: Map<string, number>;
    private deviceId: string;
    private fleetStatus: FleetStatus;
    private currentRole: FleetRole;
    private syncInterval: NodeJS.Timer | null;
    private statsInterval: NodeJS.Timer | null;
    private reconnectAttempts: number;

    constructor() {
        this.peerConnections = new Map();
        this.peerLatencies = new Map();
        this.deviceId = crypto.randomUUID();
        this.fleetStatus = FleetStatus.INACTIVE;
        this.currentRole = FleetRole.MEMBER;
        this.fleetLeaderId = null;
        this.backupLeaderId = null;
        this.syncInterval = null;
        this.statsInterval = null;
        this.reconnectAttempts = 0;

        this.networkStats = {
            averageLatency: 0,
            maxLatency: 0,
            minLatency: Number.MAX_VALUE,
            packetsLost: 0,
            bandwidth: 0,
            connectedPeers: 0,
            syncLatency: 0
        };

        this.initializeNetworkMonitoring();
    }

    /**
     * Initializes a new WebRTC peer connection with enhanced monitoring
     */
    public async initializeConnection(peerId: string): Promise<FleetMemberConnection> {
        const config = webrtcConfig.createDefaultConfig(
            this.determineRegion(),
            this.getNetworkType()
        );

        const peerConnection = await createPeerConnection(config, {
            peerConnection: null!,
            dataChannel: null!,
            lastPing: Date.now(),
            connectionQuality: 1,
            retryCount: 0
        });

        const dataChannel = await createDataChannel(peerConnection, `fleet-${peerId}`, {
            ordered: true,
            maxRetransmits: 0
        });

        const connection: FleetMemberConnection = {
            peerConnection,
            dataChannel,
            lastPing: Date.now(),
            connectionQuality: 1,
            retryCount: 0
        };

        this.setupConnectionHandlers(connection, peerId);
        this.peerConnections.set(peerId, connection);

        return connection;
    }

    /**
     * Connects to a fleet with enhanced coordination and monitoring
     */
    public async connectToFleet(fleetId: string): Promise<void> {
        if (this.fleetStatus !== FleetStatus.INACTIVE) {
            throw new Error('Already connected to a fleet');
        }

        try {
            await this.connectToSignalingServer();
            this.fleetStatus = FleetStatus.CONNECTING;

            const joinMessage: FleetSyncMessage = {
                type: FleetMessageType.MEMBER_JOIN,
                payload: {
                    deviceId: this.deviceId,
                    capabilities: this.getDeviceCapabilities()
                },
                timestamp: Date.now(),
                senderId: this.deviceId,
                sequence: 0,
                priority: 9
            };

            this.signalingConnection.send(JSON.stringify(joinMessage));
            await this.participateInLeaderElection();
            await this.establishPeerConnections();
            
            this.startSyncInterval();
            this.startStatsMonitoring();
            
            this.fleetStatus = FleetStatus.ACTIVE;
        } catch (error) {
            this.handleFleetConnectionError(error);
            throw error;
        }
    }

    /**
     * Gracefully disconnects from fleet with state preservation
     */
    public async disconnectFromFleet(): Promise<void> {
        if (this.fleetStatus === FleetStatus.INACTIVE) {
            return;
        }

        try {
            if (this.currentRole === FleetRole.LEADER) {
                await this.transferLeadership();
            }

            const leaveMessage: FleetSyncMessage = {
                type: FleetMessageType.MEMBER_LEAVE,
                payload: { deviceId: this.deviceId },
                timestamp: Date.now(),
                senderId: this.deviceId,
                sequence: 0,
                priority: 9
            };

            this.broadcastToFleet(leaveMessage);
            await this.cleanupConnections();
            
            this.stopIntervals();
            this.resetState();
        } catch (error) {
            console.error('Error during fleet disconnection:', error);
            this.forceDisconnect();
        }
    }

    /**
     * Retrieves comprehensive network performance statistics
     */
    public getNetworkStats(): FleetNetworkStats {
        return {
            ...this.networkStats,
            connectedPeers: this.peerConnections.size
        };
    }

    private async setupConnectionHandlers(connection: FleetMemberConnection, peerId: string): Promise<void> {
        connection.peerConnection.addEventListener('connectionstatechange', () => {
            this.handleConnectionStateChange(connection, peerId);
        });

        connection.dataChannel.addEventListener('message', (event) => {
            this.handleDataChannelMessage(JSON.parse(event.data), peerId);
        });

        connection.dataChannel.addEventListener('error', (error) => {
            this.handleDataChannelError(error, connection, peerId);
        });
    }

    private async participateInLeaderElection(): Promise<void> {
        const leaderScore = this.calculateLeaderScore();
        
        const electionMessage: FleetSyncMessage = {
            type: FleetMessageType.LEADER_ELECTION,
            payload: {
                deviceId: this.deviceId,
                score: leaderScore
            },
            timestamp: Date.now(),
            senderId: this.deviceId,
            sequence: 0,
            priority: 9
        };

        this.broadcastToFleet(electionMessage);
        
        return new Promise((resolve) => {
            setTimeout(() => {
                this.finalizeLeaderElection();
                resolve();
            }, LEADER_ELECTION_TIMEOUT);
        });
    }

    private calculateLeaderScore(): number {
        const latencyScore = 1 - (this.networkStats.averageLatency / MAX_LATENCY_THRESHOLD);
        const bandwidthScore = this.networkStats.bandwidth / webrtcConfig.performance.minBandwidth;
        const stabilityScore = 1 - (this.networkStats.packetsLost / 100);

        return (latencyScore + bandwidthScore + stabilityScore) / 3;
    }

    private async transferLeadership(): Promise<void> {
        if (this.backupLeaderId) {
            const transferMessage: FleetSyncMessage = {
                type: FleetMessageType.LEADER_ELECTION,
                payload: {
                    newLeaderId: this.backupLeaderId,
                    previousLeaderId: this.deviceId
                },
                timestamp: Date.now(),
                senderId: this.deviceId,
                sequence: 0,
                priority: 9
            };

            this.broadcastToFleet(transferMessage);
            await this.waitForLeadershipTransfer();
        }
    }

    private broadcastToFleet(message: FleetSyncMessage): void {
        this.peerConnections.forEach((connection) => {
            if (connection.dataChannel.readyState === 'open') {
                connection.dataChannel.send(JSON.stringify(message));
            }
        });
    }

    private startSyncInterval(): void {
        this.syncInterval = setInterval(() => {
            this.synchronizeFleetState();
        }, DEFAULT_SYNC_INTERVAL);
    }

    private startStatsMonitoring(): void {
        this.statsInterval = setInterval(async () => {
            await this.updateNetworkStats();
        }, NETWORK_STATS_INTERVAL);
    }

    private stopIntervals(): void {
        if (this.syncInterval) clearInterval(this.syncInterval);
        if (this.statsInterval) clearInterval(this.statsInterval);
    }

    private resetState(): void {
        this.fleetStatus = FleetStatus.INACTIVE;
        this.currentRole = FleetRole.MEMBER;
        this.fleetLeaderId = null;
        this.backupLeaderId = null;
        this.peerConnections.clear();
        this.peerLatencies.clear();
    }

    private async updateNetworkStats(): Promise<void> {
        const latencies: number[] = [];
        
        for (const [peerId, connection] of this.peerConnections) {
            try {
                const latency = await measureLatency(connection.dataChannel, this.networkStats);
                this.peerLatencies.set(peerId, latency);
                latencies.push(latency);
            } catch (error) {
                console.error(`Error measuring latency for peer ${peerId}:`, error);
            }
        }

        if (latencies.length > 0) {
            this.networkStats.averageLatency = latencies.reduce((a, b) => a + b) / latencies.length;
            this.networkStats.maxLatency = Math.max(...latencies);
            this.networkStats.minLatency = Math.min(...latencies);
        }
    }

    private forceDisconnect(): void {
        this.stopIntervals();
        this.cleanupConnections();
        this.resetState();
    }
}