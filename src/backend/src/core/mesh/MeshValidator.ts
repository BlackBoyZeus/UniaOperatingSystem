import { injectable } from 'inversify';
import { Logger } from 'winston'; // version: 3.8.2
import { RTCPeerConnection, RTCDataChannel } from 'webrtc'; // version: M98
import {
    MeshConfig,
    MeshPeer,
    MeshTopology,
    ValidationError,
    MAX_PEERS,
    MAX_LATENCY,
    isValidPeerCount,
    isValidLatency
} from '../../types/mesh.types';

// Global constants for validation
const VALIDATION_TIMEOUT = 5000;
const RETRY_ATTEMPTS = 3;
const BATCH_SIZE = 10;

/**
 * Result type for validation operations
 */
type ValidationResult<T = boolean> = {
    success: boolean;
    data?: T;
    error?: ValidationError;
    timestamp: number;
};

/**
 * Enhanced mesh network validator with comprehensive error handling
 */
@injectable()
export class MeshValidator {
    private readonly maxPeers: number;
    private readonly maxLatency: number;
    private readonly logger: Logger;
    private readonly validationCache: Map<string, ValidationResult>;

    constructor(config: MeshConfig, logger: Logger) {
        this.maxPeers = config.maxPeers || MAX_PEERS;
        this.maxLatency = config.maxLatency || MAX_LATENCY;
        this.logger = logger;
        this.validationCache = new Map();
    }

    /**
     * Validates entire mesh network topology
     * @param topology Current mesh network topology
     * @returns Validation result with detailed error if failed
     */
    public async validateNetwork(topology: MeshTopology): Promise<ValidationResult> {
        try {
            // Validate peer count
            const peerCount = topology.peers.size;
            if (!isValidPeerCount(peerCount)) {
                throw new ValidationError(
                    'INVALID_PEER_COUNT',
                    `Peer count ${peerCount} exceeds maximum of ${this.maxPeers}`
                );
            }

            // Validate topology structure
            const structureValidation = this.validateTopologyStructure(topology);
            if (!structureValidation.success) {
                throw structureValidation.error;
            }

            // Batch validate peer connections
            const peerValidations = await this.validatePeerConnections(Array.from(topology.peers.values()));
            const failedPeers = peerValidations.filter(result => !result.success);
            
            if (failedPeers.length > 0) {
                throw new ValidationError(
                    'PEER_VALIDATION_FAILED',
                    `${failedPeers.length} peer(s) failed validation`,
                    failedPeers
                );
            }

            return {
                success: true,
                timestamp: Date.now()
            };
        } catch (error) {
            this.logger.error('Network validation failed:', error);
            return {
                success: false,
                error: error instanceof ValidationError ? error : new ValidationError('VALIDATION_FAILED', error.message),
                timestamp: Date.now()
            };
        }
    }

    /**
     * Validates individual peer connection with enhanced WebRTC checks
     * @param peer Mesh network peer to validate
     * @returns Connection validation result
     */
    public async validatePeerConnection(peer: MeshPeer): Promise<ValidationResult> {
        try {
            // Check connection state
            if (peer.connection.connectionState !== 'connected') {
                throw new ValidationError(
                    'INVALID_CONNECTION_STATE',
                    `Peer ${peer.id} connection state: ${peer.connection.connectionState}`
                );
            }

            // Validate data channel
            if (peer.dataChannel.readyState !== 'open') {
                throw new ValidationError(
                    'INVALID_DATACHANNEL_STATE',
                    `Peer ${peer.id} data channel state: ${peer.dataChannel.readyState}`
                );
            }

            // Validate latency with timeout and retries
            const latencyValidation = await this.validateLatencyWithRetry(peer);
            if (!latencyValidation.success) {
                throw latencyValidation.error;
            }

            return {
                success: true,
                timestamp: Date.now()
            };
        } catch (error) {
            this.logger.error(`Peer ${peer.id} validation failed:`, error);
            return {
                success: false,
                error: error instanceof ValidationError ? error : new ValidationError('PEER_VALIDATION_FAILED', error.message),
                timestamp: Date.now()
            };
        }
    }

    /**
     * Validates mesh network topology structure
     * @param topology Mesh network topology
     * @returns Topology validation result
     */
    public validateTopologyStructure(topology: MeshTopology): ValidationResult {
        try {
            // Verify topology completeness
            if (!topology.peers || !topology.connections) {
                throw new ValidationError(
                    'INCOMPLETE_TOPOLOGY',
                    'Topology missing required components'
                );
            }

            // Check connection symmetry
            for (const [peerId, connections] of topology.connections) {
                for (const connectedPeerId of connections) {
                    const reverseConnections = topology.connections.get(connectedPeerId);
                    if (!reverseConnections?.includes(peerId)) {
                        throw new ValidationError(
                            'ASYMMETRIC_CONNECTIONS',
                            `Asymmetric connection between peers ${peerId} and ${connectedPeerId}`
                        );
                    }
                }
            }

            // Validate peer distribution
            const connectionCounts = Array.from(topology.connections.values())
                .map(connections => connections.length);
            
            const maxConnections = Math.max(...connectionCounts);
            const minConnections = Math.min(...connectionCounts);
            
            if (maxConnections - minConnections > 2) {
                throw new ValidationError(
                    'UNBALANCED_TOPOLOGY',
                    'Network topology is not properly balanced'
                );
            }

            return {
                success: true,
                timestamp: Date.now()
            };
        } catch (error) {
            this.logger.error('Topology structure validation failed:', error);
            return {
                success: false,
                error: error instanceof ValidationError ? error : new ValidationError('STRUCTURE_VALIDATION_FAILED', error.message),
                timestamp: Date.now()
            };
        }
    }

    /**
     * Validates peer latency with retry mechanism
     * @param peer Mesh network peer
     * @returns Latency validation result
     */
    private async validateLatencyWithRetry(peer: MeshPeer): Promise<ValidationResult> {
        for (let attempt = 0; attempt < RETRY_ATTEMPTS; attempt++) {
            try {
                const latency = peer.latency;
                if (!isValidLatency(latency)) {
                    throw new ValidationError(
                        'INVALID_LATENCY',
                        `Peer ${peer.id} latency ${latency}ms exceeds maximum ${this.maxLatency}ms`
                    );
                }

                return {
                    success: true,
                    timestamp: Date.now()
                };
            } catch (error) {
                if (attempt === RETRY_ATTEMPTS - 1) {
                    throw error;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        throw new ValidationError(
            'LATENCY_VALIDATION_FAILED',
            `Failed to validate latency after ${RETRY_ATTEMPTS} attempts`
        );
    }

    /**
     * Batch validates peer connections
     * @param peers Array of mesh peers to validate
     * @returns Array of validation results
     */
    private async validatePeerConnections(peers: MeshPeer[]): Promise<ValidationResult[]> {
        const results: ValidationResult[] = [];
        
        for (let i = 0; i < peers.length; i += BATCH_SIZE) {
            const batch = peers.slice(i, i + BATCH_SIZE);
            const batchResults = await Promise.all(
                batch.map(peer => this.validatePeerConnection(peer))
            );
            results.push(...batchResults);
        }
        
        return results;
    }
}

export { ValidationResult };