import { injectable } from 'inversify'; // v6.1.0
import AWS from 'aws-sdk'; // v2.1450.0
import { v4 as uuidv4 } from 'uuid'; // v9.0.0
import CircuitBreaker from 'opossum'; // v6.0.0
import { DatabaseConfig } from '../../config/database.config';
import { S3Service } from '../../services/storage/S3Service';
import { TaldLogger } from '../../utils/logger.utils';

// Constants for scan data management
const SCAN_TABLE = process.env.SCAN_TABLE || 'tald-lidar-scans';
const MAX_BATCH_SIZE = 25;
const SCAN_TTL_DAYS = 30;
const RETRY_ATTEMPTS = 3;
const CIRCUIT_BREAKER_TIMEOUT = 5000;

// Interfaces
interface IScanMetadata {
  deviceId: string;
  sessionId: string;
  timestamp: number;
  resolution: number;
  range: number;
  pointCount: number;
  processingLatency: number;
  tags?: Record<string, string>;
}

interface IPointCloud {
  data: Buffer;
  format: string;
  compression?: string;
}

interface SaveOptions {
  ttl?: number;
  priority?: 'high' | 'normal' | 'low';
  replication?: boolean;
  encryption?: 'AES256' | 'aws:kms';
}

interface QueryOptions {
  limit?: number;
  startKey?: AWS.DynamoDB.Key;
  consistentRead?: boolean;
}

@injectable()
export class ScanRepository {
  private dynamoClient: AWS.DynamoDB.DocumentClient;
  private s3Service: S3Service;
  private logger: TaldLogger;
  private tableName: string;
  private circuitBreaker: CircuitBreaker;

  constructor(
    private config: DatabaseConfig,
    s3Service: S3Service,
    logger: TaldLogger
  ) {
    const dbConfig = this.config.getConnectionConfig();
    this.dynamoClient = dbConfig.primary;
    this.s3Service = s3Service;
    this.logger = logger;
    this.tableName = SCAN_TABLE;

    this.circuitBreaker = new CircuitBreaker(async (operation: Function) => {
      return operation();
    }, {
      timeout: CIRCUIT_BREAKER_TIMEOUT,
      errorThresholdPercentage: 50,
      resetTimeout: 30000
    });

    this.initializeMonitoring();
  }

  private initializeMonitoring(): void {
    this.circuitBreaker.on('success', () => {
      this.logger.metric('scan_repository_success', 1);
    });

    this.circuitBreaker.on('failure', (error: Error) => {
      this.logger.error('Scan repository operation failed', error);
      this.logger.metric('scan_repository_failure', 1);
    });

    this.circuitBreaker.on('timeout', () => {
      this.logger.warn('Scan repository operation timeout');
      this.logger.metric('scan_repository_timeout', 1);
    });
  }

  public async saveScan(
    pointCloud: IPointCloud,
    metadata: IScanMetadata,
    options: SaveOptions = {}
  ): Promise<string> {
    const scanId = uuidv4();
    const timestamp = Date.now();

    try {
      // Upload point cloud data to S3
      const s3Result = await this.s3Service.uploadLidarScan(
        pointCloud.data,
        metadata.deviceId,
        {
          scanId,
          format: pointCloud.format,
          compression: pointCloud.compression,
          ...metadata
        },
        {
          contentType: 'application/octet-stream',
          encryption: options.encryption,
          multipart: true,
          tags: metadata.tags
        }
      );

      // Calculate TTL
      const ttl = Math.floor(timestamp / 1000) + (options.ttl || SCAN_TTL_DAYS * 86400);

      // Store metadata in DynamoDB
      const item = {
        scanId,
        deviceId: metadata.deviceId,
        sessionId: metadata.sessionId,
        timestamp,
        ttl,
        resolution: metadata.resolution,
        range: metadata.range,
        pointCount: metadata.pointCount,
        processingLatency: metadata.processingLatency,
        s3Key: s3Result.key,
        s3Version: s3Result.versionId,
        tags: metadata.tags
      };

      await this.circuitBreaker.fire(() => 
        this.dynamoClient.put({
          TableName: this.tableName,
          Item: item
        }).promise()
      );

      // Handle replication if enabled
      if (options.replication) {
        await this.handleReplication(scanId, item);
      }

      this.logger.info('Scan data saved successfully', {
        scanId,
        deviceId: metadata.deviceId,
        size: pointCloud.data.length,
        latency: metadata.processingLatency
      });

      return scanId;
    } catch (error) {
      this.logger.error('Failed to save scan data', error);
      throw error;
    }
  }

  public async getScanById(
    scanId: string,
    options: QueryOptions = {}
  ): Promise<{ metadata: any; pointCloud?: Buffer }> {
    try {
      // Get metadata from DynamoDB
      const result = await this.circuitBreaker.fire(() =>
        this.dynamoClient.get({
          TableName: this.tableName,
          Key: { scanId },
          ConsistentRead: options.consistentRead
        }).promise()
      );

      if (!result.Item) {
        throw new Error(`Scan not found: ${scanId}`);
      }

      // Get point cloud data from S3 if requested
      let pointCloud: Buffer | undefined;
      if (!options.limit) {
        const s3Result = await this.s3Service.getLidarScan(result.Item.s3Key);
        pointCloud = s3Result.data as Buffer;
      }

      return {
        metadata: result.Item,
        pointCloud
      };
    } catch (error) {
      this.logger.error('Failed to retrieve scan', error);
      throw error;
    }
  }

  public async getSessionScans(
    sessionId: string,
    options: QueryOptions = {}
  ): Promise<AWS.DynamoDB.DocumentClient.QueryOutput> {
    try {
      const params: AWS.DynamoDB.DocumentClient.QueryInput = {
        TableName: this.tableName,
        IndexName: 'sessionId-timestamp-index',
        KeyConditionExpression: 'sessionId = :sessionId',
        ExpressionAttributeValues: {
          ':sessionId': sessionId
        },
        Limit: options.limit,
        ExclusiveStartKey: options.startKey,
        ConsistentRead: options.consistentRead
      };

      return await this.circuitBreaker.fire(() =>
        this.dynamoClient.query(params).promise()
      );
    } catch (error) {
      this.logger.error('Failed to retrieve session scans', error);
      throw error;
    }
  }

  public async deleteScan(scanId: string): Promise<void> {
    try {
      const scan = await this.getScanById(scanId);
      
      // Delete from S3
      await this.s3Service.deleteLidarScan(scan.metadata.s3Key);

      // Delete from DynamoDB
      await this.circuitBreaker.fire(() =>
        this.dynamoClient.delete({
          TableName: this.tableName,
          Key: { scanId }
        }).promise()
      );

      this.logger.info('Scan deleted successfully', { scanId });
    } catch (error) {
      this.logger.error('Failed to delete scan', error);
      throw error;
    }
  }

  private async handleReplication(
    scanId: string,
    item: any
  ): Promise<void> {
    try {
      const replicationConfig = this.config.getReplicationConfig();
      const secondaryClients = replicationConfig.clients;

      await Promise.all(secondaryClients.map(client =>
        client.put({
          TableName: this.tableName,
          Item: item
        }).promise()
      ));

      this.logger.info('Scan replicated successfully', { scanId });
    } catch (error) {
      this.logger.error('Scan replication failed', error);
      throw error;
    }
  }
}