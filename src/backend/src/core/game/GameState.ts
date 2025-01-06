import { EventEmitter } from 'events';
import * as Automerge from 'automerge';
import {
  IGameState,
  IEnvironmentState,
  IPhysicsState,
  IStateMetrics
} from '../../interfaces/game.interface';
import {
  CRDTDocument,
  CRDTChange,
  CRDTMetrics,
  DEFAULT_SYNC_INTERVAL,
  MAX_RETRIES,
  SYNC_TIMEOUT,
  MAX_FLEET_SIZE,
  BackoffStrategy
} from '../../types/crdt.types';

// Constants for state management
const STATE_UPDATE_INTERVAL = 50; // 50ms update interval
const MAX_ENVIRONMENT_OBJECTS = 1000;
const MAX_PHYSICS_OBJECTS = 100;
const MAX_RETRY_ATTEMPTS = 3;
const SYNC_TIMEOUT_MS = 100;

/**
 * GameState class implementing CRDT-based state synchronization
 * Supports up to 32 concurrent devices with sub-50ms latency
 */
export class GameState extends EventEmitter {
  private _document: Automerge.Doc<IGameState>;
  private _changes: CRDTChange[];
  private _lastUpdate: number;
  private _metrics: IStateMetrics;
  private _retryQueue: Map<string, { attempt: number; change: CRDTChange }>;
  private _updateInterval: NodeJS.Timeout;
  private _fleetSize: number;

  constructor(gameId: string, sessionId: string, fleetSize: number) {
    super();

    if (fleetSize > MAX_FLEET_SIZE) {
      throw new Error(`Fleet size cannot exceed ${MAX_FLEET_SIZE} devices`);
    }

    // Initialize CRDT document
    this._document = Automerge.init<IGameState>({
      gameId,
      sessionId,
      fleetId: `fleet_${gameId}`,
      deviceCount: 0,
      timestamp: Date.now(),
      environment: this.initializeEnvironmentState(),
      physics: this.initializePhysicsState(),
      metrics: this.initializeMetrics()
    });

    this._changes = [];
    this._lastUpdate = Date.now();
    this._fleetSize = fleetSize;
    this._retryQueue = new Map();
    this._metrics = {
      stateUpdateLatency: 0,
      syncSuccessRate: 100,
      operationCount: 0,
      errorCount: 0
    };

    // Initialize state update interval
    this._updateInterval = setInterval(() => {
      this.processRetryQueue();
      this.validateStateConsistency();
    }, STATE_UPDATE_INTERVAL);
  }

  /**
   * Retrieves current validated game state
   */
  public getState(): IGameState {
    this.validateStateConsistency();
    return Automerge.getObjectValue(this._document);
  }

  /**
   * Updates environment state with new LiDAR data
   */
  public async updateEnvironment(environmentState: IEnvironmentState): Promise<void> {
    const startTime = Date.now();

    try {
      if (environmentState.classifiedObjects.length > MAX_ENVIRONMENT_OBJECTS) {
        throw new Error(`Environment object count exceeds maximum (${MAX_ENVIRONMENT_OBJECTS})`);
      }

      const [newDoc, change] = Automerge.change(this._document, doc => {
        doc.environment = environmentState;
        doc.timestamp = Date.now();
      });

      this._document = newDoc;
      this._changes.push({
        documentId: this._document.gameId,
        operation: 'UPDATE',
        timestamp: Date.now(),
        retryCount: 0
      });

      this.updateMetrics('environment', startTime);
      this.emit('environmentUpdated', environmentState);
    } catch (error) {
      this.handleUpdateError('environment', error);
      throw error;
    }
  }

  /**
   * Updates physics state with new simulation data
   */
  public async updatePhysics(physicsState: IPhysicsState): Promise<void> {
    const startTime = Date.now();

    try {
      if (physicsState.objects.length > MAX_PHYSICS_OBJECTS) {
        throw new Error(`Physics object count exceeds maximum (${MAX_PHYSICS_OBJECTS})`);
      }

      const [newDoc, change] = Automerge.change(this._document, doc => {
        doc.physics = physicsState;
        doc.timestamp = Date.now();
      });

      this._document = newDoc;
      this._changes.push({
        documentId: this._document.gameId,
        operation: 'UPDATE',
        timestamp: Date.now(),
        retryCount: 0
      });

      this.updateMetrics('physics', startTime);
      this.emit('physicsUpdated', physicsState);
    } catch (error) {
      this.handleUpdateError('physics', error);
      throw error;
    }
  }

  /**
   * Applies CRDT change from another device
   */
  public async applyChange(change: CRDTChange): Promise<void> {
    const startTime = Date.now();

    try {
      if (this.isChangeValid(change)) {
        const [newDoc] = Automerge.applyChanges(this._document, [change]);
        this._document = newDoc;
        this.updateMetrics('sync', startTime);
        this.emit('changeApplied', change);
      } else {
        throw new Error('Invalid change detected');
      }
    } catch (error) {
      this.handleSyncError(change, error);
      throw error;
    }
  }

  /**
   * Retrieves current performance metrics
   */
  public getMetrics(): IStateMetrics {
    return {
      ...this._metrics,
      lastUpdateLatency: Date.now() - this._lastUpdate,
      changeCount: this._changes.length,
      retryQueueSize: this._retryQueue.size
    };
  }

  /**
   * Cleans up resources on shutdown
   */
  public dispose(): void {
    clearInterval(this._updateInterval);
    this.removeAllListeners();
    this._retryQueue.clear();
    this._changes = [];
  }

  private initializeEnvironmentState(): IEnvironmentState {
    return {
      timestamp: Date.now(),
      scanQuality: 1.0,
      pointCount: 0,
      classifiedObjects: [],
      lidarMetrics: {
        scanRate: 30,
        resolution: 0.01,
        effectiveRange: 5.0,
        pointDensity: 0
      }
    };
  }

  private initializePhysicsState(): IPhysicsState {
    return {
      timestamp: Date.now(),
      objects: [],
      collisions: [],
      simulationLatency: 0
    };
  }

  private initializeMetrics(): IStateMetrics {
    return {
      stateUpdateLatency: 0,
      syncSuccessRate: 100,
      operationCount: 0,
      errorCount: 0
    };
  }

  private validateStateConsistency(): void {
    const currentState = Automerge.getObjectValue(this._document);
    
    if (!currentState || !currentState.environment || !currentState.physics) {
      throw new Error('Invalid game state detected');
    }

    if (Date.now() - currentState.timestamp > SYNC_TIMEOUT_MS) {
      this.emit('stateStale', {
        lastUpdate: currentState.timestamp,
        currentTime: Date.now()
      });
    }
  }

  private processRetryQueue(): void {
    for (const [id, entry] of this._retryQueue.entries()) {
      if (entry.attempt < MAX_RETRY_ATTEMPTS) {
        this.retryChange(entry.change);
        entry.attempt++;
      } else {
        this._retryQueue.delete(id);
        this.emit('retryFailed', entry.change);
      }
    }
  }

  private async retryChange(change: CRDTChange): Promise<void> {
    try {
      await this.applyChange(change);
      this._retryQueue.delete(change.documentId);
    } catch (error) {
      this.handleSyncError(change, error);
    }
  }

  private updateMetrics(operation: string, startTime: number): void {
    const latency = Date.now() - startTime;
    this._metrics.stateUpdateLatency = latency;
    this._metrics.operationCount++;
    this._lastUpdate = Date.now();
  }

  private handleUpdateError(type: string, error: Error): void {
    this._metrics.errorCount++;
    this._metrics.syncSuccessRate = 
      ((this._metrics.operationCount - this._metrics.errorCount) / 
       this._metrics.operationCount) * 100;
    
    this.emit('updateError', {
      type,
      error: error.message,
      timestamp: Date.now()
    });
  }

  private handleSyncError(change: CRDTChange, error: Error): void {
    this._metrics.errorCount++;
    this._retryQueue.set(change.documentId, {
      attempt: 1,
      change
    });
    
    this.emit('syncError', {
      change,
      error: error.message,
      timestamp: Date.now()
    });
  }

  private isChangeValid(change: CRDTChange): boolean {
    return (
      change &&
      change.documentId === this._document.gameId &&
      change.timestamp > this._lastUpdate &&
      change.retryCount < MAX_RETRY_ATTEMPTS
    );
  }
}