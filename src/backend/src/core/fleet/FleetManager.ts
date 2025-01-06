import { injectable, singleton } from 'tsyringe';
import { EventEmitter } from 'events'; // v3.3.0
import * as Automerge from 'automerge'; // v2.0
import { CircuitBreaker } from 'opossum'; // v6.0.0
import * as compression from 'compression'; // v1.7.4

import { FleetState } from './FleetState';
import { FleetSync } from './FleetSync';
import {
  IFleet,
  IFleetState,
  IFleetMember,
  IMeshTopology,
  FleetRole,
  FleetStatus,
  MeshTopologyType,
  IFleetNetworkStats,
  IFleetSecurity
} from '../../interfaces/fleet.interface';

import {
  CRDTDocument,
  CRDTChange,
  CRDTOperation,
  FleetPerformanceMetrics,
  SyncError,
  MAX_FLEET_SIZE,
  DEFAULT_SYNC_INTERVAL
} from '../../types/crdt.types';

/**
 * Enhanced FleetManager class for managing fleet operations with advanced monitoring
 * and reliability features. Supports up to 32 devices with <50ms latency.
 */
@injectable()
@singleton()
export class FleetManager extends EventEmitter {
  private readonly fleetState: FleetState;
  private readonly fleetSync: FleetSync;
  private readonly activeMembers: Map<string, IFleetMember>;
  private readonly networkBreaker: CircuitBreaker;
  private readonly metricsCollector: Map<string, FleetPerformanceMetrics>;
  private readonly connectionPool: Map<string, RTCPeerConnection>;
  private readonly retryQueue: Map<string, CRDTChange>;

  private static readonly MAX_FLEET_SIZE = MAX_FLEET_SIZE;
  private static readonly SYNC_INTERVAL = DEFAULT_SYNC_INTERVAL;
  private static readonly CIRCUIT_BREAKER_TIMEOUT = 5000;
  private static readonly MAX_RETRY_ATTEMPTS = 3;

  constructor(
    fleetConfig: IFleet,
    private readonly securityConfig: IFleetSecurity
  ) {
    super();
    this.fleetState = new FleetState(fleetConfig);
    this.fleetSync = new FleetSync(fleetConfig);
    this.activeMembers = new Map();
    this.metricsCollector = new Map();
    this.connectionPool = new Map();
    this.retryQueue = new Map();

    // Initialize circuit breaker for network operations
    this.networkBreaker = new CircuitBreaker(this.executeNetworkOperation.bind(this), {
      timeout: FleetManager.CIRCUIT_BREAKER_TIMEOUT,
      resetTimeout: FleetManager.CIRCUIT_BREAKER_TIMEOUT * 2,
      errorThresholdPercentage: 50
    });

    this.initializeEventHandlers();
    this.startMetricsCollection();
  }

  /**
   * Creates a new fleet with enhanced monitoring and security
   */
  public async createFleet(fleetConfig: IFleet): Promise<IFleet> {
    try {
      await this.validateFleetConfig(fleetConfig);

      const fleet = await this.networkBreaker.fire(async () => {
        const newFleet = await this.fleetState.addMember({
          ...fleetConfig,
          role: FleetRole.LEADER,
          status: FleetStatus.ACTIVE,
          joinedAt: Date.now()
        });

        await this.fleetSync.startSync();
        return newFleet;
      });

      this.emit('fleetCreated', { fleetId: fleet.id, timestamp: Date.now() });
      return fleet;

    } catch (error) {
      this.handleError('createFleet', error);
      throw error;
    }
  }

  /**
   * Adds a new member to the fleet with validation and monitoring
   */
  public async joinFleet(memberId: string, capabilities: any): Promise<void> {
    try {
      if (this.activeMembers.size >= FleetManager.MAX_FLEET_SIZE) {
        throw new Error(`Fleet size limit (${FleetManager.MAX_FLEET_SIZE}) reached`);
      }

      await this.validateMemberCapabilities(capabilities);

      const member: IFleetMember = {
        id: memberId,
        role: FleetRole.MEMBER,
        status: FleetStatus.CONNECTING,
        joinedAt: Date.now(),
        lastActive: Date.now(),
        capabilities
      };

      await this.networkBreaker.fire(async () => {
        await this.fleetState.addMember(member);
        await this.fleetSync.handleMemberAdded(member);
        this.activeMembers.set(memberId, member);
      });

      this.startMemberMonitoring(memberId);
      this.emit('memberJoined', { memberId, timestamp: Date.now() });

    } catch (error) {
      this.handleError('joinFleet', error);
      throw error;
    }
  }

  /**
   * Synchronizes fleet state with error handling and retry mechanism
   */
  public async synchronizeFleet(change: CRDTChange): Promise<void> {
    try {
      await this.networkBreaker.fire(async () => {
        await this.fleetSync.handleStateChange(change);
      });

    } catch (error) {
      if (change.retryCount < FleetManager.MAX_RETRY_ATTEMPTS) {
        change.retryCount = (change.retryCount || 0) + 1;
        this.retryQueue.set(change.documentId, change);
        this.scheduleRetry(change);
      } else {
        this.handleError('synchronizeFleet', error);
        throw error;
      }
    }
  }

  /**
   * Removes a member from the fleet with cleanup
   */
  public async leaveFleet(memberId: string): Promise<void> {
    try {
      await this.networkBreaker.fire(async () => {
        await this.fleetState.removeMember(memberId);
        await this.fleetSync.handleMemberRemoved(memberId);
        
        this.activeMembers.delete(memberId);
        this.metricsCollector.delete(memberId);
        this.cleanupConnections(memberId);
      });

      this.emit('memberLeft', { memberId, timestamp: Date.now() });

    } catch (error) {
      this.handleError('leaveFleet', error);
      throw error;
    }
  }

  /**
   * Returns current fleet state and metrics
   */
  public getFleetState(): IFleetState {
    return this.fleetState.getState();
  }

  /**
   * Returns performance metrics for the fleet
   */
  public getMetrics(): FleetPerformanceMetrics {
    const fleetMetrics = this.fleetState.getMetrics();
    const syncMetrics = this.fleetSync.getPerformanceMetrics();

    return {
      averageLatency: syncMetrics.averageLatency,
      syncSuccessRate: syncMetrics.syncSuccessRate,
      memberCount: this.activeMembers.size,
      lastUpdateTimestamp: Date.now()
    };
  }

  /**
   * Validates fleet configuration
   */
  private async validateFleetConfig(config: IFleet): Promise<void> {
    if (!config.id || !config.name) {
      throw new Error('Invalid fleet configuration');
    }

    if (config.maxDevices > FleetManager.MAX_FLEET_SIZE) {
      throw new Error(`Maximum fleet size is ${FleetManager.MAX_FLEET_SIZE}`);
    }
  }

  /**
   * Validates member capabilities
   */
  private async validateMemberCapabilities(capabilities: any): Promise<void> {
    if (!capabilities.lidarSupport) {
      throw new Error('LiDAR support required for fleet membership');
    }

    if (capabilities.networkBandwidth < 1000) { // 1000 Kbps minimum
      throw new Error('Insufficient network bandwidth');
    }
  }

  /**
   * Initializes event handlers for fleet management
   */
  private initializeEventHandlers(): void {
    this.fleetSync.on('stateChanged', this.handleStateChange.bind(this));
    this.fleetSync.on('syncError', this.handleSyncError.bind(this));
    this.fleetSync.on('peerConnected', this.handlePeerConnection.bind(this));
    this.networkBreaker.on('timeout', this.handleNetworkTimeout.bind(this));
  }

  /**
   * Starts metrics collection for fleet monitoring
   */
  private startMetricsCollection(): void {
    setInterval(() => {
      const metrics = this.getMetrics();
      this.emit('metricsUpdated', metrics);
    }, 1000);
  }

  /**
   * Starts monitoring for individual fleet members
   */
  private startMemberMonitoring(memberId: string): void {
    const monitor = setInterval(() => {
      const member = this.activeMembers.get(memberId);
      if (member) {
        const latency = Date.now() - member.lastActive;
        if (latency > FleetManager.CIRCUIT_BREAKER_TIMEOUT) {
          this.handleMemberTimeout(memberId);
        }
      } else {
        clearInterval(monitor);
      }
    }, FleetManager.SYNC_INTERVAL);
  }

  /**
   * Executes network operations with timeout protection
   */
  private async executeNetworkOperation<T>(operation: () => Promise<T>): Promise<T> {
    return new Promise(async (resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Network operation timeout'));
      }, FleetManager.CIRCUIT_BREAKER_TIMEOUT);

      try {
        const result = await operation();
        clearTimeout(timeout);
        resolve(result);
      } catch (error) {
        clearTimeout(timeout);
        reject(error);
      }
    });
  }

  /**
   * Schedules retry for failed operations
   */
  private scheduleRetry(change: CRDTChange): void {
    setTimeout(async () => {
      try {
        await this.synchronizeFleet(change);
        this.retryQueue.delete(change.documentId);
      } catch (error) {
        // Retry exhausted, handled in synchronizeFleet
      }
    }, Math.pow(2, change.retryCount) * 1000);
  }

  /**
   * Cleans up connections for departing members
   */
  private cleanupConnections(memberId: string): void {
    const connection = this.connectionPool.get(memberId);
    if (connection) {
      connection.close();
      this.connectionPool.delete(memberId);
    }
  }

  /**
   * Handles state changes in the fleet
   */
  private handleStateChange(change: CRDTChange): void {
    this.emit('stateChanged', {
      change,
      timestamp: Date.now()
    });
  }

  /**
   * Handles synchronization errors
   */
  private handleSyncError(error: SyncError): void {
    this.emit('syncError', {
      error,
      timestamp: Date.now()
    });
  }

  /**
   * Handles peer connection events
   */
  private handlePeerConnection(peerId: string): void {
    const member = this.activeMembers.get(peerId);
    if (member) {
      member.status = FleetStatus.ACTIVE;
      member.lastActive = Date.now();
    }
  }

  /**
   * Handles network timeouts
   */
  private handleNetworkTimeout(): void {
    this.emit('networkTimeout', {
      timestamp: Date.now(),
      activeMembers: this.activeMembers.size
    });
  }

  /**
   * Handles member timeouts
   */
  private handleMemberTimeout(memberId: string): void {
    const member = this.activeMembers.get(memberId);
    if (member) {
      member.status = FleetStatus.INACTIVE;
      this.emit('memberTimeout', {
        memberId,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Generic error handler with logging and events
   */
  private handleError(context: string, error: Error): void {
    this.emit('error', {
      context,
      error: error.message,
      timestamp: Date.now()
    });
  }

  /**
   * Cleans up resources on shutdown
   */
  public dispose(): void {
    this.fleetSync.dispose();
    for (const connection of this.connectionPool.values()) {
      connection.close();
    }
    this.activeMembers.clear();
    this.metricsCollector.clear();
    this.connectionPool.clear();
    this.retryQueue.clear();
  }
}