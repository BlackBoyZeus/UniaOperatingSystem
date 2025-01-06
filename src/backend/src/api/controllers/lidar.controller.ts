import { injectable } from 'inversify'; // version: 6.0.1
import { 
  controller, 
  httpPost, 
  httpGet, 
  request, 
  response, 
  authorize 
} from 'inversify-express-utils'; // version: 6.4.3
import { Request, Response } from 'express'; // version: 4.18.2
import { Logger } from 'winston'; // version: 3.10.0
import { rateLimit } from 'express-rate-limit'; // version: 6.9.0
import { MetricsService } from '@metrics/service'; // version: 1.0.0

import { LidarService } from '../../services/lidar/LidarService';
import {
  ILidarConfig,
  IPointCloud,
  IScanMetadata,
  ScanQuality,
  ProcessingMode,
  IFleetState
} from '../../interfaces/lidar.interface';

// Constants for rate limiting and validation
const MAX_SCAN_RATE = 30;
const MIN_RESOLUTION = 0.01;
const MAX_RANGE = 5.0;
const MAX_PROCESSING_TIME = 50;
const MAX_FLEET_SIZE = 32;
const MAX_FLEET_LATENCY = 50;

@injectable()
@controller('/api/lidar')
@authorize('lidar-access')
export class LidarController {
  constructor(
    private readonly lidarService: LidarService,
    private readonly metricsService: MetricsService,
    private readonly logger: Logger
  ) {}

  /**
   * Process raw LiDAR scan data with real-time performance monitoring
   * @param req Request containing raw scan data and processing parameters
   * @returns Processed point cloud with metadata
   */
  @httpPost('/scan')
  @rateLimit({ windowMs: 1000, max: 30 }) // Enforce 30Hz scan rate
  public async processScan(
    @request() req: Request,
    @response() res: Response
  ): Promise<Response> {
    const processingStart = performance.now();

    try {
      // Validate request body
      const { scanData, config, fleetId } = req.body;
      this.validateScanRequest(scanData, config);

      // Track processing metrics
      const metricTags = {
        deviceId: req.headers['x-device-id'] as string,
        fleetId,
        processingMode: config.processingMode
      };
      this.metricsService.startTimer('lidar_processing', metricTags);

      // Process scan with performance monitoring
      const processedScan = await this.lidarService.processScan(
        scanData,
        config as ILidarConfig
      );

      // Validate scan quality
      const qualityValidation = await this.lidarService.validateScanQuality(
        processedScan as IPointCloud
      );

      if (!qualityValidation.isValid) {
        throw new Error(`Scan quality validation failed: ${qualityValidation.errors?.join(', ')}`);
      }

      // Update fleet state if part of fleet
      if (fleetId) {
        await this.updateFleetState(fleetId, processedScan);
      }

      // Calculate and validate processing time
      const processingTime = performance.now() - processingStart;
      if (processingTime > MAX_PROCESSING_TIME) {
        this.logger.warn('Processing time exceeded threshold', {
          processingTime,
          threshold: MAX_PROCESSING_TIME,
          ...metricTags
        });
      }

      // Record final metrics
      this.metricsService.recordMetric('lidar_processing_time', processingTime, metricTags);
      this.metricsService.recordMetric('lidar_point_count', processedScan.points.length, metricTags);
      this.metricsService.recordMetric('lidar_quality_score', qualityValidation.confidence, metricTags);

      return res.status(200).json({
        pointCloud: processedScan,
        metadata: {
          processingTime,
          quality: qualityValidation,
          fleetId,
          timestamp: Date.now()
        }
      });

    } catch (error) {
      this.logger.error('LiDAR scan processing failed', {
        error: error.message,
        stack: error.stack,
        deviceId: req.headers['x-device-id'],
        processingTime: performance.now() - processingStart
      });

      this.metricsService.incrementCounter('lidar_processing_errors');

      return res.status(500).json({
        error: 'Scan processing failed',
        message: error.message
      });
    }
  }

  /**
   * Update fleet state with latest scan data
   * @param req Request containing fleet state update
   * @returns Updated fleet state
   */
  @httpPost('/fleet/state')
  @rateLimit({ windowMs: 1000, max: 20 })
  public async updateFleetState(
    @request() req: Request,
    @response() res: Response
  ): Promise<Response> {
    const updateStart = performance.now();

    try {
      const { fleetId, deviceId, scanMetadata } = req.body;

      // Validate fleet membership and size
      const currentFleetState = await this.lidarService.getFleetState(fleetId);
      if (currentFleetState.devices.length >= MAX_FLEET_SIZE) {
        throw new Error(`Fleet size limit exceeded (max: ${MAX_FLEET_SIZE})`);
      }

      // Update fleet state
      const updatedState = await this.lidarService.updateFleetState({
        fleetId,
        deviceId,
        scanMetadata,
        timestamp: Date.now()
      });

      // Record metrics
      const updateTime = performance.now() - updateStart;
      this.metricsService.recordMetric('fleet_state_update_time', updateTime, {
        fleetId,
        deviceId
      });

      if (updateTime > MAX_FLEET_LATENCY) {
        this.logger.warn('Fleet state update exceeded latency threshold', {
          updateTime,
          threshold: MAX_FLEET_LATENCY,
          fleetId
        });
      }

      return res.status(200).json({
        fleetState: updatedState,
        updateTime
      });

    } catch (error) {
      this.logger.error('Fleet state update failed', {
        error: error.message,
        stack: error.stack,
        fleetId: req.body.fleetId,
        updateTime: performance.now() - updateStart
      });

      this.metricsService.incrementCounter('fleet_state_update_errors');

      return res.status(500).json({
        error: 'Fleet state update failed',
        message: error.message
      });
    }
  }

  private validateScanRequest(scanData: Buffer, config: ILidarConfig): void {
    if (!scanData || !config) {
      throw new Error('Missing required scan data or configuration');
    }

    if (config.scanRate > MAX_SCAN_RATE) {
      throw new Error(`Invalid scan rate: ${config.scanRate}Hz (max: ${MAX_SCAN_RATE}Hz)`);
    }

    if (config.resolution < MIN_RESOLUTION) {
      throw new Error(`Invalid resolution: ${config.resolution}cm (min: ${MIN_RESOLUTION}cm)`);
    }

    if (config.range > MAX_RANGE) {
      throw new Error(`Invalid range: ${config.range}m (max: ${MAX_RANGE}m)`);
    }
  }

  private async updateFleetState(fleetId: string, scanData: IPointCloud): Promise<void> {
    const fleetUpdate: IFleetState = {
      fleetId,
      deviceId: scanData.deviceId,
      timestamp: Date.now(),
      scanMetadata: {
        quality: scanData.quality,
        pointCount: scanData.points.length,
        confidence: scanData.confidence
      }
    };

    await this.lidarService.updateFleetState(fleetUpdate);
  }
}