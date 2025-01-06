import { injectable, inject } from 'inversify'; // version: 6.0.1
import { v4 as uuidv4 } from 'uuid'; // version: 9.0.0

import {
    ILidarConfig,
    IPointCloud,
    ProcessingMode,
    ScanQuality,
    IScanMetadata,
    ILidarProcessor,
    IErrorHandler,
    IResourceMonitor
} from '../../interfaces/lidar.interface';

import { PointCloudGenerator } from './PointCloudGenerator';
import { ScanOptimizer } from './ScanOptimizer';

// Global constants for processing constraints
const PROCESSING_TIMEOUT_MS = 50;
const SCAN_RATE_HZ = 30;
const MIN_RESOLUTION_CM = 0.01;
const MAX_RANGE_M = 5.0;
const MAX_GPU_TEMP_C = 85;
const MIN_BATTERY_PERCENT = 10;
const ERROR_RETRY_ATTEMPTS = 3;

@injectable()
export class LidarProcessor implements ILidarProcessor {
    private processingStartTime: number;
    private lastProcessingMode: ProcessingMode;
    private consecutiveErrors: number = 0;
    private isProcessing: boolean = false;

    constructor(
        @inject('ILidarConfig') private readonly config: ILidarConfig,
        @inject('PointCloudGenerator') private readonly pointCloudGenerator: PointCloudGenerator,
        @inject('ScanOptimizer') private readonly scanOptimizer: ScanOptimizer,
        @inject('IErrorHandler') private readonly errorHandler: IErrorHandler,
        @inject('IResourceMonitor') private readonly resourceMonitor: IResourceMonitor
    ) {
        this.validateConfiguration();
        this.lastProcessingMode = config.processingMode;
    }

    /**
     * Processes raw point cloud data with comprehensive error handling and adaptive processing
     * @param rawScanData Buffer containing raw LiDAR scan data
     * @returns Promise<IPointCloud> Processed and optimized point cloud
     * @throws Error if processing fails or exceeds timeout
     */
    public async processPointCloud(rawScanData: Buffer): Promise<IPointCloud> {
        if (this.isProcessing) {
            throw new Error('Processing already in progress');
        }

        this.isProcessing = true;
        this.processingStartTime = performance.now();

        try {
            // Validate system resources
            await this.validateSystemResources();

            // Determine optimal processing mode
            const processingMode = await this.determineProcessingMode();

            // Generate initial point cloud
            const pointCloud = await this.pointCloudGenerator.generatePointCloud(rawScanData);

            // Validate initial results
            const initialValidation = await this.validateScan(pointCloud);
            if (!initialValidation.isValid) {
                throw new Error(`Initial point cloud validation failed: ${initialValidation.errors?.join(', ')}`);
            }

            // Optimize point cloud
            const optimizedCloud = await this.scanOptimizer.optimizeScan(pointCloud, processingMode);

            // Validate optimization results
            const finalValidation = await this.validateScan(optimizedCloud.pointCloud);
            if (!finalValidation.isValid) {
                throw new Error(`Optimization validation failed: ${finalValidation.errors?.join(', ')}`);
            }

            // Generate and attach metadata
            const metadata = this.generateScanMetadata(optimizedCloud.pointCloud, performance.now() - this.processingStartTime);

            this.consecutiveErrors = 0;
            return {
                ...optimizedCloud.pointCloud,
                metadata
            };

        } catch (error) {
            this.consecutiveErrors++;
            await this.handleProcessingError(error);
            throw error;

        } finally {
            this.isProcessing = false;
            await this.updateProcessingMetrics();
        }
    }

    /**
     * Validates scan results against quality thresholds
     * @param pointCloud Processed point cloud to validate
     * @returns Validation result with confidence score
     */
    public async validateScan(pointCloud: IPointCloud): Promise<{
        isValid: boolean;
        confidence: number;
        errors?: string[];
    }> {
        const errors: string[] = [];
        let confidence = 1.0;

        // Validate point count
        const pointCount = this.getPointCount(pointCloud);
        if (pointCount < 1000) {
            errors.push(`Insufficient point count: ${pointCount}`);
            confidence *= 0.5;
        }

        // Validate processing time
        const processingTime = performance.now() - this.processingStartTime;
        if (processingTime > PROCESSING_TIMEOUT_MS) {
            errors.push(`Processing time exceeded: ${processingTime}ms`);
            confidence *= 0.7;
        }

        // Validate point cloud quality
        if (pointCloud.quality === ScanQuality.LOW) {
            errors.push('Low scan quality detected');
            confidence *= 0.8;
        }

        // Validate point cloud density
        if (pointCloud.density < 1000) {
            errors.push(`Low point density: ${pointCloud.density}`);
            confidence *= 0.9;
        }

        return {
            isValid: errors.length === 0,
            confidence,
            errors: errors.length > 0 ? errors : undefined
        };
    }

    /**
     * Generates comprehensive metadata for scan operation
     * @param pointCloud Processed point cloud
     * @param processingTime Processing duration
     * @returns Scan metadata
     */
    public generateScanMetadata(pointCloud: IPointCloud, processingTime: number): IScanMetadata {
        return {
            scanId: uuidv4(),
            timestamp: Date.now(),
            processingTime,
            quality: pointCloud.quality,
            errorRate: 1 - pointCloud.confidence,
            powerConsumption: this.resourceMonitor.getCurrentPowerDraw(),
            systemMetrics: {
                gpuTemperature: this.resourceMonitor.getGPUTemperature(),
                memoryUsage: this.resourceMonitor.getMemoryUsage(),
                batteryLevel: this.resourceMonitor.getBatteryLevel()
            }
        };
    }

    /**
     * Optimizes scan rate based on current system conditions
     * @param currentConfig Current LiDAR configuration
     * @param systemLoad Current system load
     * @returns Optimized scan rate
     */
    public async optimizeScanRate(currentConfig: ILidarConfig, systemLoad: number): Promise<number> {
        const maxRate = SCAN_RATE_HZ;
        const gpuTemp = this.resourceMonitor.getGPUTemperature();
        const batteryLevel = this.resourceMonitor.getBatteryLevel();

        // Reduce scan rate under high load or temperature
        if (systemLoad > 0.8 || gpuTemp > MAX_GPU_TEMP_C) {
            return Math.max(maxRate * 0.5, 15);
        }

        // Reduce scan rate on low battery
        if (batteryLevel < MIN_BATTERY_PERCENT) {
            return Math.max(maxRate * 0.7, 20);
        }

        return maxRate;
    }

    private async validateSystemResources(): Promise<void> {
        const gpuTemp = this.resourceMonitor.getGPUTemperature();
        const batteryLevel = this.resourceMonitor.getBatteryLevel();
        const memoryUsage = this.resourceMonitor.getMemoryUsage();

        if (gpuTemp > MAX_GPU_TEMP_C) {
            throw new Error(`GPU temperature too high: ${gpuTemp}°C`);
        }

        if (batteryLevel < MIN_BATTERY_PERCENT) {
            throw new Error(`Battery level too low: ${batteryLevel}%`);
        }

        if (memoryUsage > 0.9) {
            throw new Error(`Memory usage too high: ${memoryUsage * 100}%`);
        }
    }

    private async determineProcessingMode(): Promise<ProcessingMode> {
        const systemLoad = this.resourceMonitor.getSystemLoad();
        const batteryLevel = this.resourceMonitor.getBatteryLevel();

        if (batteryLevel < MIN_BATTERY_PERCENT || systemLoad > 0.8) {
            return ProcessingMode.POWER_SAVE;
        }

        if (this.consecutiveErrors > 0) {
            return ProcessingMode.HIGH_QUALITY;
        }

        return this.config.processingMode;
    }

    private async handleProcessingError(error: Error): Promise<void> {
        await this.errorHandler.logError({
            component: 'LidarProcessor',
            error,
            context: {
                consecutiveErrors: this.consecutiveErrors,
                processingMode: this.lastProcessingMode,
                processingTime: performance.now() - this.processingStartTime
            }
        });

        if (this.consecutiveErrors >= ERROR_RETRY_ATTEMPTS) {
            await this.errorHandler.triggerErrorRecovery();
        }
    }

    private validateConfiguration(): void {
        if (this.config.scanRate > SCAN_RATE_HZ) {
            throw new Error(`Invalid scan rate: ${this.config.scanRate}Hz (max: ${SCAN_RATE_HZ}Hz)`);
        }

        if (this.config.resolution < MIN_RESOLUTION_CM) {
            throw new Error(`Invalid resolution: ${this.config.resolution}cm (min: ${MIN_RESOLUTION_CM}cm)`);
        }

        if (this.config.range > MAX_RANGE_M) {
            throw new Error(`Invalid range: ${this.config.range}m (max: ${MAX_RANGE_M}m)`);
        }
    }

    private getPointCount(pointCloud: IPointCloud): number {
        return pointCloud.points.length / (3 * Float32Array.BYTES_PER_ELEMENT);
    }

    private async updateProcessingMetrics(): Promise<void> {
        const processingTime = performance.now() - this.processingStartTime;
        await this.resourceMonitor.recordMetrics({
            component: 'LidarProcessor',
            processingTime,
            processingMode: this.lastProcessingMode,
            errorCount: this.consecutiveErrors
        });
    }
}