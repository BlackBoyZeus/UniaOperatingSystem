// External imports - version specified for security tracking
import * as Automerge from 'automerge'; // @version ^2.0.0

// Internal imports
import { 
    IFleet, 
    IFleetMember, 
    IFleetConnection, 
    IFleetSchema,
    IFleetMemberSchema 
} from '../interfaces/fleet.interface';
import WebRTCService from './webrtc.service';
import ApiService from './api.service';
import { 
    FleetStatus, 
    FleetRole, 
    FleetMessageType,
    MAX_FLEET_SIZE,
    DEFAULT_SYNC_INTERVAL,
    MAX_LATENCY_THRESHOLD
} from '../types/fleet.types';

// Service constants
const SYNC_INTERVAL = DEFAULT_SYNC_INTERVAL; // 50ms sync interval
const RECONNECT_TIMEOUT = 5000;
const LEADER_FAILOVER_TIMEOUT = 3000;
const MAX_RETRY_ATTEMPTS = 3;
const NETWORK_QUALITY_THRESHOLD = 0.8;

/**
 * Enhanced service class managing fleet operations with advanced monitoring,
 * failover, and state synchronization capabilities.
 */
export class FleetService {
    private currentFleet: IFleet | null = null;
    private connections: Map<string, IFleetConnection> = new Map();
    private networkMetrics: Map<string, number> = new Map();
    private fleetState: Automerge.Doc<IFleet>;
    private syncInterval: NodeJS.Timer | null = null;
    private monitorInterval: NodeJS.Timer | null = null;
    private retryAttempts: Map<string, number> = new Map();
    private backupLeaders: string[] = [];

    constructor(
        private webrtcService: WebRTCService,
        private apiService: ApiService
    ) {
        this.fleetState = Automerge.init<IFleet>();
        this.setupErrorHandling();
    }

    /**
     * Creates a new fleet with advanced monitoring and failover capabilities
     * @param fleetName Name of the fleet
     * @param maxDevices Maximum number of devices (up to 32)
     */
    public async createFleet(fleetName: string, maxDevices: number = MAX_FLEET_SIZE): Promise<IFleet> {
        try {
            // Validate fleet parameters
            if (maxDevices > MAX_FLEET_SIZE) {
                throw new Error(`Fleet size cannot exceed ${MAX_FLEET_SIZE} devices`);
            }

            // Create fleet through API
            const fleetData = await this.apiService.request<IFleet>({
                url: '/fleet/create',
                method: 'POST',
                data: { name: fleetName, maxDevices }
            });

            // Validate fleet data
            const validatedFleet = IFleetSchema.parse(fleetData);

            // Initialize CRDT state
            this.fleetState = Automerge.change(this.fleetState, 'Initialize fleet', doc => {
                doc.id = validatedFleet.id;
                doc.name = validatedFleet.name;
                doc.maxDevices = validatedFleet.maxDevices;
                doc.members = [];
                doc.status = FleetStatus.ACTIVE;
                doc.networkStats = {
                    averageLatency: 0,
                    maxLatency: 0,
                    minLatency: Number.MAX_VALUE,
                    packetsLost: 0,
                    bandwidth: 0,
                    connectedPeers: 0,
                    syncLatency: 0
                };
            });

            // Initialize WebRTC connections
            await this.initializeFleetConnections(validatedFleet);

            // Start monitoring and sync intervals
            this.startFleetMonitoring();
            this.startStateSync();

            this.currentFleet = validatedFleet;
            return validatedFleet;

        } catch (error) {
            console.error('Failed to create fleet:', error);
            throw error;
        }
    }

    /**
     * Joins an existing fleet with state synchronization
     * @param fleetId ID of the fleet to join
     */
    public async joinFleet(fleetId: string): Promise<void> {
        try {
            // Validate current state
            if (this.currentFleet) {
                throw new Error('Already part of a fleet');
            }

            // Join fleet through API
            const fleetData = await this.apiService.request<IFleet>({
                url: '/fleet/join',
                method: 'POST',
                data: { fleetId }
            });

            // Validate fleet data
            const validatedFleet = IFleetSchema.parse(fleetData);

            // Initialize WebRTC connections
            await this.initializeFleetConnections(validatedFleet);

            // Sync initial state
            await this.syncFleetState();

            // Start monitoring
            this.startFleetMonitoring();

            this.currentFleet = validatedFleet;

        } catch (error) {
            console.error('Failed to join fleet:', error);
            throw error;
        }
    }

    /**
     * Leaves current fleet with graceful connection termination
     */
    public async leaveFleet(): Promise<void> {
        if (!this.currentFleet) {
            return;
        }

        try {
            // Notify other members
            await this.broadcastLeave();

            // Clean up connections
            await this.cleanupConnections();

            // Update API
            await this.apiService.request({
                url: '/fleet/leave',
                method: 'POST',
                data: { fleetId: this.currentFleet.id }
            });

            this.stopMonitoring();
            this.currentFleet = null;

        } catch (error) {
            console.error('Error leaving fleet:', error);
            throw error;
        }
    }

    /**
     * Synchronizes fleet state using CRDT
     */
    public async syncFleetState(): Promise<void> {
        if (!this.currentFleet) {
            return;
        }

        try {
            const syncStart = performance.now();

            // Get latest state from all peers
            const peerStates = await Promise.all(
                Array.from(this.connections.entries()).map(async ([peerId, connection]) => {
                    try {
                        const state = await this.webrtcService.sendGameState(connection.peerConnection);
                        return { peerId, state };
                    } catch (error) {
                        console.error(`Failed to get state from peer ${peerId}:`, error);
                        return null;
                    }
                })
            );

            // Merge states using CRDT
            const validStates = peerStates.filter(state => state !== null);
            if (validStates.length > 0) {
                this.fleetState = validStates.reduce((merged, { state }) => {
                    return Automerge.merge(merged, state as Automerge.Doc<IFleet>);
                }, this.fleetState);
            }

            // Update sync metrics
            const syncLatency = performance.now() - syncStart;
            this.updateNetworkMetrics('syncLatency', syncLatency);

        } catch (error) {
            console.error('Fleet state sync failed:', error);
            throw error;
        }
    }

    /**
     * Handles fleet leader failover process
     */
    private async handleLeaderFailover(): Promise<void> {
        if (!this.currentFleet) {
            return;
        }

        try {
            // Select new leader based on network metrics
            const newLeader = this.selectNewLeader();
            if (!newLeader) {
                throw new Error('No suitable leader found');
            }

            // Update fleet state
            this.fleetState = Automerge.change(this.fleetState, 'Update leader', doc => {
                const member = doc.members.find(m => m.id === newLeader);
                if (member) {
                    member.role = FleetRole.LEADER;
                }
            });

            // Broadcast leader change
            await this.broadcastLeaderChange(newLeader);

            // Update backup leaders
            this.updateBackupLeaders();

        } catch (error) {
            console.error('Leader failover failed:', error);
            throw error;
        }
    }

    /**
     * Monitors fleet health and network quality
     */
    private async monitorFleetHealth(): Promise<void> {
        if (!this.currentFleet) {
            return;
        }

        try {
            // Check all connections
            for (const [peerId, connection] of this.connections.entries()) {
                const metrics = await this.webrtcService.monitorNetworkQuality(connection.peerConnection);
                
                // Update metrics
                this.networkMetrics.set(peerId, metrics.connectionQuality);

                // Check for degraded connections
                if (metrics.connectionQuality < NETWORK_QUALITY_THRESHOLD) {
                    await this.handleDegradedConnection(peerId);
                }
            }

            // Update fleet status
            this.updateFleetStatus();

        } catch (error) {
            console.error('Fleet health monitoring failed:', error);
            throw error;
        }
    }

    private async initializeFleetConnections(fleet: IFleet): Promise<void> {
        for (const member of fleet.members) {
            if (member.id !== this.apiService.deviceId) {
                const connection = await this.webrtcService.initializeConnection(member.id);
                this.connections.set(member.id, connection);
            }
        }
    }

    private startFleetMonitoring(): void {
        this.monitorInterval = setInterval(() => {
            this.monitorFleetHealth().catch(error => {
                console.error('Fleet monitoring error:', error);
            });
        }, 1000);
    }

    private startStateSync(): void {
        this.syncInterval = setInterval(() => {
            this.syncFleetState().catch(error => {
                console.error('State sync error:', error);
            });
        }, SYNC_INTERVAL);
    }

    private stopMonitoring(): void {
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
        }
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
        }
    }

    private setupErrorHandling(): void {
        process.on('unhandledRejection', (error) => {
            console.error('Unhandled promise rejection in FleetService:', error);
            this.handleServiceError(error);
        });
    }

    private async handleServiceError(error: any): Promise<void> {
        console.error('FleetService error:', error);
        
        if (this.currentFleet) {
            try {
                await this.leaveFleet();
            } catch (cleanupError) {
                console.error('Error during fleet cleanup:', cleanupError);
            }
        }
    }
}

export default FleetService;