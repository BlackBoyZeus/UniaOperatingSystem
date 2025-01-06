import { EventEmitter } from 'events'; // latest
import { WebSocket } from 'ws'; // ^8.13.0
import axios from 'axios'; // ^1.4.0
import { Buffer } from 'buffer'; // ^6.0.3

import {
  Point3D,
  PointCloudData,
  ScanQuality,
  ScanConfig,
  ScanMetadata,
  ScanState,
  FleetSyncState,
  PointCloudDataSchema,
  ScanConfigSchema
} from '../types/lidar.types';

import {
  validateScanConfig,
  parsePointCloudBuffer,
  calculateScanQuality,
  optimizePointCloud,
  calculateProcessingMetrics
} from '../utils/lidar.utils';

import {
  LIDAR_SCAN_SETTINGS,
  LIDAR_VISUALIZATION,
  LIDAR_PERFORMANCE
} from '../constants/lidar.constants';

/**
 * Service class for managing LiDAR operations in the web frontend
 * Handles real-time scanning visualization and fleet synchronization
 * @version 1.0.0
 */
export class LidarService {
  private readonly eventEmitter: EventEmitter;
  private currentState: ScanState;
  private wsConnection: WebSocket | null;
  private updateInterval: NodeJS.Timer | null;
  private syncInterval: NodeJS.Timer | null;
  private pointCloudBuffer: Buffer;
  private scanHistory: Map<string, ScanMetadata>;
  private processingMetrics: {
    lastProcessingTime: number;
    averageLatency: number;
    pointsProcessed: number;
  };

  constructor(initialConfig: ScanConfig) {
    this.eventEmitter = new EventEmitter();
    this.wsConnection = null;
    this.updateInterval = null;
    this.syncInterval = null;
    this.pointCloudBuffer = Buffer.alloc(LIDAR_PERFORMANCE.BUFFER_SIZE);
    this.scanHistory = new Map();
    this.processingMetrics = {
      lastProcessingTime: 0,
      averageLatency: 0,
      pointsProcessed: 0
    };

    this.currentState = {
      isActive: false,
      currentScan: null,
      metadata: null
    };

    // Validate initial configuration
    if (!validateScanConfig(initialConfig)) {
      throw new Error('Invalid LiDAR scan configuration');
    }
  }

  /**
   * Starts real-time LiDAR scanning with fleet synchronization
   * @param config - Scan configuration parameters
   * @returns Promise resolving when scanning starts
   */
  public async startScanning(config: ScanConfig): Promise<void> {
    try {
      // Validate configuration
      const validatedConfig = ScanConfigSchema.parse(config);

      // Initialize WebSocket connection
      await this.initializeWebSocket();

      // Start scan update interval (30Hz)
      this.updateInterval = setInterval(() => {
        this.processScanUpdate();
      }, LIDAR_PERFORMANCE.UPDATE_INTERVAL);

      // Update state
      this.currentState = {
        ...this.currentState,
        isActive: true
      };

      this.eventEmitter.emit('scanStarted', {
        timestamp: Date.now(),
        config: validatedConfig
      });
    } catch (error) {
      console.error('Failed to start scanning:', error);
      throw error;
    }
  }

  /**
   * Processes incoming point cloud data with performance optimization
   * @param rawData - Binary point cloud data
   * @returns Processed point cloud data
   */
  public async processPointCloud(rawData: Buffer): Promise<PointCloudData> {
    const startTime = performance.now();

    try {
      // Validate data size
      if (rawData.length > LIDAR_PERFORMANCE.BUFFER_SIZE) {
        throw new Error('Point cloud data exceeds buffer size');
      }

      // Parse point cloud
      const points = parsePointCloudBuffer(rawData);

      // Optimize for visualization
      const optimizedPoints = optimizePointCloud(
        points,
        LIDAR_PERFORMANCE.MAX_POINTS_PER_SCAN
      );

      // Calculate processing metrics
      const metrics = calculateProcessingMetrics(startTime, points.length);

      // Update metrics
      this.processingMetrics = {
        lastProcessingTime: metrics.processingTime,
        averageLatency: (this.processingMetrics.averageLatency + metrics.processingTime) / 2,
        pointsProcessed: points.length
      };

      // Validate against schema
      const pointCloudData: PointCloudData = {
        points: optimizedPoints,
        timestamp: Date.now(),
        rawData: rawData
      };

      const validated = PointCloudDataSchema.parse(pointCloudData);

      // Emit metrics if processing time exceeds threshold
      if (!metrics.withinLatencyTarget) {
        this.eventEmitter.emit('processingLatencyWarning', metrics);
      }

      return validated;
    } catch (error) {
      console.error('Point cloud processing error:', error);
      throw error;
    }
  }

  /**
   * Stops LiDAR scanning and cleans up resources
   */
  public async stopScanning(): Promise<void> {
    try {
      // Clear intervals
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
        this.updateInterval = null;
      }

      if (this.syncInterval) {
        clearInterval(this.syncInterval);
        this.syncInterval = null;
      }

      // Close WebSocket connection
      if (this.wsConnection) {
        this.wsConnection.close();
        this.wsConnection = null;
      }

      // Update state
      this.currentState = {
        ...this.currentState,
        isActive: false,
        currentScan: null
      };

      this.eventEmitter.emit('scanStopped', {
        timestamp: Date.now(),
        finalMetrics: this.processingMetrics
      });
    } catch (error) {
      console.error('Failed to stop scanning:', error);
      throw error;
    }
  }

  /**
   * Subscribes to LiDAR scan events
   * @param event - Event name
   * @param callback - Event handler
   */
  public subscribe(event: string, callback: (data: any) => void): void {
    this.eventEmitter.on(event, callback);
  }

  /**
   * Unsubscribes from LiDAR scan events
   * @param event - Event name
   * @param callback - Event handler
   */
  public unsubscribe(event: string, callback: (data: any) => void): void {
    this.eventEmitter.off(event, callback);
  }

  /**
   * Initializes WebSocket connection with retry logic
   */
  private async initializeWebSocket(): Promise<void> {
    const connect = () => {
      this.wsConnection = new WebSocket('ws://localhost:8080/lidar');

      this.wsConnection.on('open', () => {
        console.log('WebSocket connection established');
      });

      this.wsConnection.on('message', async (data: Buffer) => {
        try {
          const processedData = await this.processPointCloud(data);
          this.currentState = {
            ...this.currentState,
            currentScan: processedData
          };
          this.eventEmitter.emit('scanUpdate', processedData);
        } catch (error) {
          console.error('Failed to process WebSocket message:', error);
        }
      });

      this.wsConnection.on('close', () => {
        console.log('WebSocket connection closed, retrying...');
        setTimeout(connect, 1000);
      });

      this.wsConnection.on('error', (error) => {
        console.error('WebSocket error:', error);
      });
    };

    connect();
  }

  /**
   * Processes scan updates at 30Hz
   */
  private processScanUpdate(): void {
    if (!this.currentState.isActive || !this.wsConnection) {
      return;
    }

    try {
      // Send scan request
      this.wsConnection.send(JSON.stringify({
        type: 'scanRequest',
        timestamp: Date.now()
      }));
    } catch (error) {
      console.error('Failed to process scan update:', error);
    }
  }
}