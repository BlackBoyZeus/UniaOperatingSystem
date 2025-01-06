import { injectable } from 'inversify';
import { Logger } from 'winston'; // version: 3.8.2
import { RTCPeerConnection, RTCDataChannel } from 'webrtc'; // version: M98
import {
    MeshConfig,
    MeshPeer,
    MeshTopology,
    MeshMetrics,
    MAX_PEERS,
    MAX_LATENCY,
    isValidLatency,
    ConnectionStats
} from '../../types/mesh.types';
import { MeshValidator } from './MeshValidator';

// Constants for optimization parameters
const OPTIMIZATION_INTERVAL = 1000;
const PERFORMANCE_HISTORY_SIZE = 100;
const BOTTLENECK_THRESHOLD = 0.8;
const LATENCY_WEIGHT = 0.4;
const STABILITY_WEIGHT = 0.3;
const DISTRIBUTION_WEIGHT = 0.3;

/**
 * Network performance score interface
 */
interface NetworkScore {
    overall: number;
    latencyScore: number;
    stabilityScore: number;
    distributionScore: number;
    timestamp: number;
}

/**
 * Bottleneck analysis result interface
 */
interface BottleneckAnalysis {
    overloadedPeers: string[];
    congestionPoints: Map<string, number>;
    predictedBottlenecks: string[];
    recommendations: string[];
    timestamp: number;
}

/**
 * Optimization result interface
 */
interface OptimizationResult {
    success: boolean;
    score: NetworkScore;
    bottlenecks: BottleneckAnalysis;
    appliedStrategies: string[];
    timestamp: number;
}

/**
 * Enhanced mesh network optimizer implementing advanced performance monitoring
 * and adaptive optimization strategies
 */
@injectable()
export class MeshOptimizer {
    private readonly validator: MeshValidator;
    private readonly maxPeers: number;
    private readonly maxLatency: number;
    private readonly optimizationInterval: number;
    private performanceHistory: Array<MeshMetrics>;
    private readonly logger: Logger;

    constructor(config: MeshConfig, logger: Logger) {
        this.validator = new MeshValidator(config, logger);
        this.maxPeers = config.maxPeers || MAX_PEERS;
        this.maxLatency = config.maxLatency || MAX_LATENCY;
        this.optimizationInterval = OPTIMIZATION_INTERVAL;
        this.performanceHistory = [];
        this.logger = logger;
    }

    /**
     * Calculates comprehensive network performance score
     * @param topology Current mesh network topology
     * @param historicalMetrics Historical performance metrics
     * @returns Detailed network performance score
     */
    private calculateNetworkScore(
        topology: MeshTopology,
        historicalMetrics: Array<MeshMetrics>
    ): NetworkScore {
        // Calculate latency score
        const latencies = Array.from(topology.peers.values()).map(peer => peer.latency);
        const avgLatency = latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length;
        const latencyScore = Math.max(0, 1 - (avgLatency / this.maxLatency));

        // Calculate stability score
        const connectionStates = Array.from(topology.peers.values())
            .map(peer => peer.connection.connectionState === 'connected' ? 1 : 0);
        const stabilityScore = connectionStates.reduce((sum, state) => sum + state, 0) / connectionStates.length;

        // Calculate distribution score
        const connections = Array.from(topology.connections.values());
        const avgConnections = connections.reduce((sum, conns) => sum + conns.length, 0) / connections.length;
        const variance = connections.reduce((sum, conns) => sum + Math.pow(conns.length - avgConnections, 2), 0) / connections.length;
        const distributionScore = Math.max(0, 1 - (variance / Math.pow(this.maxPeers, 2)));

        // Calculate weighted overall score
        const overall = (
            latencyScore * LATENCY_WEIGHT +
            stabilityScore * STABILITY_WEIGHT +
            distributionScore * DISTRIBUTION_WEIGHT
        );

        return {
            overall,
            latencyScore,
            stabilityScore,
            distributionScore,
            timestamp: Date.now()
        };
    }

    /**
     * Identifies current and potential network bottlenecks
     * @param topology Current mesh network topology
     * @param historicalMetrics Historical performance metrics
     * @returns Detailed bottleneck analysis
     */
    private identifyBottlenecks(
        topology: MeshTopology,
        historicalMetrics: Array<MeshMetrics>
    ): BottleneckAnalysis {
        const overloadedPeers: string[] = [];
        const congestionPoints = new Map<string, number>();
        const predictedBottlenecks: string[] = [];
        const recommendations: string[] = [];

        // Analyze current load distribution
        for (const [peerId, peer] of topology.peers) {
            const connectionCount = topology.connections.get(peerId)?.length || 0;
            const loadFactor = connectionCount / this.maxPeers;

            if (loadFactor > BOTTLENECK_THRESHOLD) {
                overloadedPeers.push(peerId);
                congestionPoints.set(peerId, loadFactor);
            }

            // Predict future bottlenecks based on latency trends
            if (peer.latency > this.maxLatency * 0.7) {
                predictedBottlenecks.push(peerId);
                recommendations.push(`Consider redistributing connections from peer ${peerId}`);
            }
        }

        return {
            overloadedPeers,
            congestionPoints,
            predictedBottlenecks,
            recommendations,
            timestamp: Date.now()
        };
    }

    /**
     * Optimizes network topology based on current performance metrics
     * @param topology Current mesh network topology
     * @returns Optimization result with detailed metrics
     */
    public async optimizeNetwork(topology: MeshTopology): Promise<OptimizationResult> {
        try {
            // Validate current topology
            const validationResult = await this.validator.validateNetwork(topology);
            if (!validationResult.success) {
                throw new Error('Network validation failed');
            }

            // Calculate current network score
            const currentScore = this.calculateNetworkScore(topology, this.performanceHistory);
            const bottlenecks = this.identifyBottlenecks(topology, this.performanceHistory);

            // Apply optimization strategies if needed
            const appliedStrategies: string[] = [];
            if (currentScore.overall < 0.8 || bottlenecks.overloadedPeers.length > 0) {
                const rebalanceResult = await this.rebalanceConnections(topology, bottlenecks);
                if (rebalanceResult.success) {
                    appliedStrategies.push('connection_rebalancing');
                }
            }

            // Update performance history
            this.updatePerformanceHistory(currentScore);

            return {
                success: true,
                score: currentScore,
                bottlenecks,
                appliedStrategies,
                timestamp: Date.now()
            };
        } catch (error) {
            this.logger.error('Network optimization failed:', error);
            throw error;
        }
    }

    /**
     * Rebalances peer connections to optimize network performance
     * @param topology Current mesh network topology
     * @param bottlenecks Current bottleneck analysis
     * @returns Success status of rebalancing operation
     */
    private async rebalanceConnections(
        topology: MeshTopology,
        bottlenecks: BottleneckAnalysis
    ): Promise<{ success: boolean }> {
        try {
            for (const overloadedPeerId of bottlenecks.overloadedPeers) {
                const connections = topology.connections.get(overloadedPeerId) || [];
                const targetPeers = Array.from(topology.peers.keys())
                    .filter(id => !bottlenecks.overloadedPeers.includes(id))
                    .filter(id => (topology.connections.get(id)?.length || 0) < this.maxPeers * 0.7);

                if (targetPeers.length > 0) {
                    // Redistribute connections
                    const connectionsToMove = connections.slice(
                        Math.floor(this.maxPeers * 0.7)
                    );
                    
                    for (let i = 0; i < connectionsToMove.length; i++) {
                        const targetPeer = targetPeers[i % targetPeers.length];
                        const connectionToMove = connectionsToMove[i];
                        
                        // Update connection mappings
                        topology.connections.get(overloadedPeerId)?.splice(
                            topology.connections.get(overloadedPeerId)?.indexOf(connectionToMove) || 0,
                            1
                        );
                        topology.connections.get(targetPeer)?.push(connectionToMove);
                    }
                }
            }

            return { success: true };
        } catch (error) {
            this.logger.error('Connection rebalancing failed:', error);
            return { success: false };
        }
    }

    /**
     * Updates performance history with new metrics
     * @param score Current network score
     */
    private updatePerformanceHistory(score: NetworkScore): void {
        this.performanceHistory.push({
            timestamp: score.timestamp,
            overall: score.overall,
            latency: score.latencyScore,
            stability: score.stabilityScore,
            distribution: score.distributionScore
        });

        if (this.performanceHistory.length > PERFORMANCE_HISTORY_SIZE) {
            this.performanceHistory.shift();
        }
    }

    /**
     * Monitors network performance and triggers optimization if needed
     * @param topology Current mesh network topology
     */
    public async monitorPerformance(topology: MeshTopology): Promise<void> {
        try {
            const currentScore = this.calculateNetworkScore(topology, this.performanceHistory);
            
            if (currentScore.overall < 0.8) {
                await this.optimizeNetwork(topology);
            }

            this.updatePerformanceHistory(currentScore);
        } catch (error) {
            this.logger.error('Performance monitoring failed:', error);
            throw error;
        }
    }
}

export { NetworkScore, BottleneckAnalysis, OptimizationResult };
export { calculateNetworkScore, identifyBottlenecks } from './MeshOptimizer';