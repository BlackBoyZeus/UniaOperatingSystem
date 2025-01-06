import { EventEmitter } from 'events'; // v3.3.0
import * as Automerge from 'automerge'; // v2.0

import {
  IFleet,
  IFleetState,
  IFleetMember,
  IMeshTopology,
  FleetRole,
  FleetStatus,
  MeshTopologyType
} from '../../interfaces/fleet.interface';

import {
  CRDTDocument,
  CRDTChange,
  CRDTOperation,
  CRDTSyncConfig,
  BackoffStrategy,
  FleetPerformanceMetrics,
  SyncError,
  SyncStats,
  DEFAULT_SYNC_INTERVAL,
  MAX_RETRIES,
  MAX_FLEET_SIZE,
  MAX_LATENCY_THRESHOLD
} from '../../types/crdt.types';

/**
 * Core class implementing CRDT-based fleet state management
 * Handles state synchronization for up to 32 devices with enhanced error handling
 */
export class FleetState extends EventEmitter {
  private document: Automerge.Doc<IFleetState>;
  private members: Map<string, IFleetMember>;
  private topology: IMeshTopology;
  private syncConfig: CRDTSyncConfig;
  private metrics: FleetPerformanceMetrics;
  private retryQueue: Map<string, CRDTChange>;
  private version: number;
  private readonly maxFleetSize: number = MAX_FLEET_SIZE;
  private readonly syncInterval: number = DEFAULT_SYNC_INTERVAL;

  constructor(
    fleetConfig: IFleet,
    syncConfig: CRDTSyncConfig = {
      syncInterval: DEFAULT_SYNC_INTERVAL,
      maxRetries: MAX_RETRIES,
      timeout: 1000,
      latencyThreshold: MAX_LATENCY_THRESHOLD,
      backoffStrategy: BackoffStrategy.EXPONENTIAL
    }
  ) {
    super();
    this.document = Automerge.init();
    this.members = new Map();
    this.retryQueue = new Map();
    this.version = 0;
    this.syncConfig = syncConfig;
    
    this.initializeState(fleetConfig);
    this.setupPerformanceMonitoring();
    this.startSyncLoop();
  }

  /**
   * Initializes the fleet state with configuration and validation
   */
  private initializeState(fleetConfig: IFleet): void {
    this.document = Automerge.change(this.document, 'Initialize fleet state', doc => {
      doc.gameState = Automerge.init();
      doc.environmentState = Automerge.init();
      doc.syncTimestamp = Date.now();
      doc.stateVersion = 0;
      doc.pendingChanges = [];
    });

    this.topology = {
      type: MeshTopologyType.HYBRID,
      connections: new Map(),
      health: 1.0
    };

    this.metrics = {
      averageLatency: 0,
      syncSuccessRate: 1.0,
      memberCount: 0,
      lastUpdateTimestamp: Date.now()
    };
  }

  /**
   * Adds a new member to the fleet with validation and error handling
   */
  public async addMember(member: IFleetMember): Promise<void> {
    try {
      if (this.members.size >= this.maxFleetSize) {
        throw new Error(`Fleet size limit (${this.maxFleetSize}) reached`);
      }

      await this.validateMember(member);

      this.document = Automerge.change(this.document, `Add member ${member.id}`, doc => {
        if (!doc.pendingChanges) doc.pendingChanges = [];
        doc.pendingChanges.push({
          type: CRDTOperation.INSERT,
          target: 'members',
          value: member
        });
      });

      this.members.set(member.id, member);
      this.updateTopology();
      this.emit('memberAdded', { memberId: member.id, timestamp: Date.now() });

    } catch (error) {
      this.handleError('addMember', error, member);
      throw error;
    }
  }

  /**
   * Synchronizes fleet state with error handling and retry mechanism
   */
  public async synchronizeState(change: CRDTChange): Promise<void> {
    const startTime = Date.now();
    try {
      await this.validateChange(change);

      const [newDoc, patch] = Automerge.applyChanges(this.document, [change]);
      this.document = newDoc;
      this.version++;

      this.updateMetrics({
        syncLatency: Date.now() - startTime,
        operation: change.operation
      });

      this.emit('stateChanged', {
        version: this.version,
        timestamp: Date.now(),
        change: patch
      });

    } catch (error) {
      await this.handleSyncError(change, error);
    }
  }

  /**
   * Validates member capabilities and requirements
   */
  private async validateMember(member: IFleetMember): Promise<void> {
    if (!member.capabilities.lidarSupport) {
      throw new Error('LiDAR support required for fleet membership');
    }

    if (member.capabilities.networkBandwidth < 1000) { // 1000 Kbps minimum
      throw new Error('Insufficient network bandwidth');
    }

    const existingMember = this.members.get(member.id);
    if (existingMember) {
      throw new Error(`Member ${member.id} already exists in fleet`);
    }
  }

  /**
   * Validates CRDT changes before application
   */
  private async validateChange(change: CRDTChange): Promise<void> {
    if (!change.documentId || !change.operation) {
      throw new Error('Invalid change format');
    }

    if (change.retryCount > this.syncConfig.maxRetries) {
      throw new Error('Maximum retry attempts exceeded');
    }

    const latency = Date.now() - change.timestamp;
    if (latency > this.syncConfig.latencyThreshold) {
      throw new Error(`Change latency (${latency}ms) exceeds threshold`);
    }
  }

  /**
   * Updates mesh network topology
   */
  private updateTopology(): void {
    this.topology.connections.clear();
    
    for (const [id, member] of this.members) {
      const connections = new Set<string>();
      
      for (const [otherId, otherMember] of this.members) {
        if (id !== otherId && this.canConnect(member, otherMember)) {
          connections.add(otherId);
        }
      }
      
      this.topology.connections.set(id, connections);
    }

    this.topology.health = this.calculateTopologyHealth();
  }

  /**
   * Determines if two members can establish a connection
   */
  private canConnect(member1: IFleetMember, member2: IFleetMember): boolean {
    const distance = this.calculateDistance(member1.position, member2.position);
    return distance <= Math.min(member1.capabilities.maxRange, member2.capabilities.maxRange);
  }

  /**
   * Calculates 3D distance between positions
   */
  private calculateDistance(pos1: IPosition, pos2: IPosition): number {
    return Math.sqrt(
      Math.pow(pos2.x - pos1.x, 2) +
      Math.pow(pos2.y - pos1.y, 2) +
      Math.pow(pos2.z - pos1.z, 2)
    );
  }

  /**
   * Calculates mesh topology health score
   */
  private calculateTopologyHealth(): number {
    if (this.members.size === 0) return 1.0;

    const expectedConnections = (this.members.size * (this.members.size - 1)) / 2;
    let actualConnections = 0;

    for (const connections of this.topology.connections.values()) {
      actualConnections += connections.size;
    }

    return Math.min(actualConnections / expectedConnections, 1.0);
  }

  /**
   * Sets up performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    setInterval(() => {
      this.metrics.lastUpdateTimestamp = Date.now();
      this.emit('metricsUpdated', this.metrics);
    }, 5000);
  }

  /**
   * Starts the state synchronization loop
   */
  private startSyncLoop(): void {
    setInterval(async () => {
      try {
        for (const [id, change] of this.retryQueue) {
          await this.synchronizeState(change);
          this.retryQueue.delete(id);
        }
      } catch (error) {
        this.handleError('syncLoop', error);
      }
    }, this.syncInterval);
  }

  /**
   * Handles synchronization errors with retry mechanism
   */
  private async handleSyncError(change: CRDTChange, error: Error): Promise<void> {
    const syncError: SyncError = {
      documentId: change.documentId,
      operation: change.operation,
      timestamp: Date.now(),
      error: error.message,
      retryCount: (change.retryCount || 0) + 1
    };

    if (syncError.retryCount <= this.syncConfig.maxRetries) {
      change.retryCount = syncError.retryCount;
      this.retryQueue.set(change.documentId, change);
    }

    this.emit('syncError', syncError);
  }

  /**
   * Updates performance metrics
   */
  private updateMetrics(data: { syncLatency: number; operation: CRDTOperation }): void {
    this.metrics.averageLatency = (
      this.metrics.averageLatency * 0.9 + data.syncLatency * 0.1
    );
    this.metrics.memberCount = this.members.size;
    this.metrics.lastUpdateTimestamp = Date.now();
  }

  /**
   * Generic error handler with logging and events
   */
  private handleError(context: string, error: Error, data?: any): void {
    this.emit('error', {
      context,
      error: error.message,
      timestamp: Date.now(),
      data
    });
  }

  /**
   * Returns current fleet state metrics
   */
  public getMetrics(): FleetPerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Returns current topology state
   */
  public getTopology(): IMeshTopology {
    return { ...this.topology };
  }
}