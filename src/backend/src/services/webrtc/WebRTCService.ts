import { injectable } from 'inversify'; // version: 6.0.1
import { EventEmitter } from 'events'; // version: 3.3.0

import { webrtcConfig, meshNetworkConfig } from '../../config/webrtc.config';
import { SignalingService } from './SignalingService';
import {
    MeshConfig,
    MeshPeer,
    MeshTopology,
    MeshEventType,
    MeshEvent,
    PerformanceMetrics
} from '../../types/mesh.types';

// Constants for WebRTC service configuration
const PING_INTERVAL = 1000;
const LATENCY_THRESHOLD = 50;
const TOPOLOGY_OPTIMIZATION_INTERVAL = 5000;
const DATA_CHANNEL_LABEL = 'tald-mesh';
const PERFORMANCE_SAMPLING_RATE = 100;
const ERROR_RETRY_ATTEMPTS = 3;
const BANDWIDTH_THRESHOLD = 5000000; // 5 Mbps

/**
 * Enhanced WebRTC service for TALD UNIA platform's P2P mesh networking
 * Manages peer connections, data channels, and fleet-based multiplayer gaming
 * with advanced performance monitoring and topology optimization
 */
@injectable()
export class WebRTCService {
    private peers: Map<string, MeshPeer>;
    private topology: MeshTopology;
    private eventEmitter: EventEmitter;
    private performanceMonitor: PerformanceMonitor;
    private topologyOptimizer: TopologyOptimizer;
    private errorHandler: ErrorHandler;

    constructor(
        private readonly config: MeshConfig,
        private readonly signalingService: SignalingService
    ) {
        this.peers = new Map();
        this.eventEmitter = new EventEmitter();
        this.initializeService();
    }

    /**
     * Initializes the WebRTC service with monitoring and optimization components
     */
    private initializeService(): void {
        this.topology = {
            peers: new Map(),
            connections: new Map(),
            environmentData: new Map(),
            createdAt: Date.now(),
            lastUpdated: Date.now()
        };

        this.performanceMonitor = new PerformanceMonitor({
            samplingRate: PERFORMANCE_SAMPLING_RATE,
            latencyThreshold: LATENCY_THRESHOLD,
            bandwidthThreshold: BANDWIDTH_THRESHOLD
        });

        this.topologyOptimizer = new TopologyOptimizer({
            maxPeers: this.config.maxPeers,
            optimizationInterval: TOPOLOGY_OPTIMIZATION_INTERVAL
        });

        this.errorHandler = new ErrorHandler({
            maxRetries: ERROR_RETRY_ATTEMPTS,
            retryBackoff: 'exponential'
        });

        this.setupEventHandlers();
        this.startPerformanceMonitoring();
    }

    /**
     * Creates a new WebRTC peer connection with enhanced monitoring
     * @param peerId Unique identifier for the peer
     * @returns Promise resolving to the established peer connection
     */
    public async createPeerConnection(peerId: string): Promise<RTCPeerConnection> {
        try {
            const peerConnection = new RTCPeerConnection(webrtcConfig);
            
            // Set up data channel with optimized configuration
            const dataChannel = peerConnection.createDataChannel(DATA_CHANNEL_LABEL, {
                ordered: true,
                maxRetransmits: 3,
                maxPacketLifeTime: 1000
            });

            // Initialize peer monitoring
            const peer: MeshPeer = {
                id: peerId,
                connection: peerConnection,
                dataChannel,
                latency: 0,
                connectionState: 'new',
                lastSeen: Date.now()
            };

            // Set up connection state monitoring
            peerConnection.onconnectionstatechange = () => {
                this.handleConnectionStateChange(peer);
            };

            peerConnection.onicecandidate = (event) => {
                if (event.candidate) {
                    this.signalingService.handleSignaling({
                        type: 'ice-candidate',
                        candidate: event.candidate,
                        peerId
                    }, peerId);
                }
            };

            // Monitor data channel state
            dataChannel.onopen = () => this.handleDataChannelOpen(peer);
            dataChannel.onclose = () => this.handleDataChannelClose(peer);
            dataChannel.onerror = (error) => this.handleDataChannelError(peer, error);

            // Set up performance monitoring
            this.setupPeerMonitoring(peer);

            this.peers.set(peerId, peer);
            this.topology.peers.set(peerId, peer);

            return peerConnection;

        } catch (error) {
            this.errorHandler.handleError('PEER_CONNECTION_ERROR', error, { peerId });
            throw error;
        }
    }

    /**
     * Monitors peer performance metrics and triggers optimization if needed
     * @param peerId Identifier of the peer to monitor
     */
    private async monitorPeerPerformance(peerId: string): Promise<void> {
        const peer = this.peers.get(peerId);
        if (!peer) return;

        try {
            const stats = await peer.connection.getStats();
            const metrics: PerformanceMetrics = {
                averageLatency: 0,
                syncSuccessRate: 0,
                meshStability: 0,
                peerConnections: this.peers.size,
                dataChannelStats: {
                    bytesReceived: 0,
                    bytesSent: 0,
                    packetsLost: 0,
                    roundTripTime: 0
                }
            };

            stats.forEach(report => {
                if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                    metrics.averageLatency = report.currentRoundTripTime * 1000;
                    metrics.meshStability = this.calculateMeshStability(report);
                }
            });

            this.performanceMonitor.updateMetrics(peerId, metrics);

            // Check if optimization is needed
            if (this.shouldOptimizeTopology(metrics)) {
                await this.optimizeTopology();
            }

        } catch (error) {
            this.errorHandler.handleError('PERFORMANCE_MONITORING_ERROR', error, { peerId });
        }
    }

    /**
     * Optimizes mesh network topology based on performance metrics
     */
    private async optimizeTopology(): Promise<void> {
        try {
            const currentMetrics = this.performanceMonitor.getAggregateMetrics();
            const optimizedTopology = this.topologyOptimizer.optimize(
                this.topology,
                currentMetrics
            );

            // Apply topology changes
            await this.applyTopologyChanges(optimizedTopology);

            // Broadcast updated topology
            await this.signalingService.broadcastTopologyUpdate(
                optimizedTopology,
                { immediate: true }
            );

            this.topology = optimizedTopology;

        } catch (error) {
            this.errorHandler.handleError('TOPOLOGY_OPTIMIZATION_ERROR', error);
        }
    }

    /**
     * Handles WebRTC errors with automatic recovery attempts
     * @param error Error object
     * @param context Error context information
     */
    private async handleError(error: Error, context: ErrorContext): Promise<void> {
        try {
            const errorType = this.errorHandler.classifyError(error);
            const recovery = await this.errorHandler.attemptRecovery(errorType, context);

            if (recovery.success) {
                this.performanceMonitor.recordRecovery(context.peerId);
            } else {
                // Escalate unrecoverable errors
                this.emitMeshEvent({
                    type: MeshEventType.PEER_DISCONNECTED,
                    peerId: context.peerId,
                    data: { error: error.message },
                    timestamp: Date.now()
                });
            }

        } catch (recoveryError) {
            // Log critical errors
            console.error('Critical error in error handler:', recoveryError);
        }
    }

    /**
     * Emits mesh network events with performance data
     * @param event Mesh network event
     */
    private emitMeshEvent(event: MeshEvent): void {
        const enrichedEvent = {
            ...event,
            performance: this.performanceMonitor.getMetrics(event.peerId),
            topology: this.topologyOptimizer.getTopologyStatus()
        };

        this.eventEmitter.emit(event.type, enrichedEvent);
    }
}

export default WebRTCService;