import AWS from 'aws-sdk'; // v2.1450.0
import { v4 as uuidv4 } from 'uuid'; // v9.0.0
import retry from 'retry'; // v0.13.0
import CircuitBreaker from 'opossum'; // v6.0.0
import { RateLimiter } from 'limiter'; // v2.0.0
import { injectable } from 'inversify'; // v6.1.0
import { AWSConfig } from '../../config/aws.config';
import { TaldLogger } from '../../utils/logger.utils';
import { Readable } from 'stream';

// Global constants
const LIDAR_BUCKET = process.env.LIDAR_BUCKET_NAME || 'tald-lidar-data';
const ASSETS_BUCKET = process.env.ASSETS_BUCKET_NAME || 'tald-game-assets';
const MAX_UPLOAD_SIZE = 1024 * 1024 * 50; // 50MB
const MULTIPART_THRESHOLD = 1024 * 1024 * 5; // 5MB
const MAX_RETRY_ATTEMPTS = 3;
const RATE_LIMIT_RPS = 100;

// Types
interface UploadOptions {
  contentType?: string;
  encryption?: 'AES256' | 'aws:kms';
  tags?: Record<string, string>;
  expiresIn?: number;
  multipart?: boolean;
}

interface GetOptions {
  streaming?: boolean;
  range?: { start: number; end: number };
  ifModifiedSince?: Date;
}

interface UploadResult {
  key: string;
  url: string;
  etag: string;
  versionId?: string;
  metadata: Record<string, string>;
}

interface GetResult {
  data: Buffer | Readable;
  metadata: Record<string, string>;
  contentType: string;
  contentLength: number;
  lastModified: Date;
}

@injectable()
export class S3Service {
  private s3Client: AWS.S3;
  private logger: TaldLogger;
  private circuitBreaker: CircuitBreaker;
  private rateLimiter: RateLimiter;

  constructor(
    private awsConfig: AWSConfig,
    logger: TaldLogger
  ) {
    const s3Config = this.awsConfig.getS3Config();
    this.s3Client = s3Config['primary'];
    this.logger = logger;

    // Initialize rate limiter
    this.rateLimiter = new RateLimiter({
      tokensPerInterval: RATE_LIMIT_RPS,
      interval: 'second'
    });

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker(async (operation: Function) => {
      await this.rateLimiter.removeTokens(1);
      return operation();
    }, {
      timeout: 30000,
      errorThresholdPercentage: 50,
      resetTimeout: 30000
    });

    this.setupMonitoring();
  }

  private setupMonitoring(): void {
    this.circuitBreaker.on('success', () => {
      this.logger.metric('s3_operation_success', 1);
    });

    this.circuitBreaker.on('failure', (error) => {
      this.logger.error('S3 operation failed', error);
      this.logger.metric('s3_operation_failure', 1);
    });

    this.circuitBreaker.on('timeout', () => {
      this.logger.warn('S3 operation timeout');
      this.logger.metric('s3_operation_timeout', 1);
    });
  }

  private async executeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
    const operation_retry = retry.operation({
      retries: MAX_RETRY_ATTEMPTS,
      factor: 2,
      minTimeout: 1000,
      maxTimeout: 5000
    });

    return new Promise((resolve, reject) => {
      operation_retry.attempt(async (currentAttempt) => {
        try {
          const result = await this.circuitBreaker.fire(() => operation());
          resolve(result);
        } catch (error) {
          if (operation_retry.retry(error)) {
            this.logger.warn(`Retrying S3 operation, attempt ${currentAttempt}`, { error });
            return;
          }
          reject(operation_retry.mainError());
        }
      });
    });
  }

  public async uploadLidarScan(
    scanData: Buffer | Readable,
    deviceId: string,
    metadata: Record<string, string>,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    const key = `scans/${deviceId}/${uuidv4()}`;
    const uploadParams: AWS.S3.PutObjectRequest = {
      Bucket: LIDAR_BUCKET,
      Key: key,
      Body: scanData,
      ContentType: options.contentType || 'application/octet-stream',
      Metadata: {
        ...metadata,
        deviceId,
        timestamp: new Date().toISOString()
      },
      ServerSideEncryption: options.encryption || 'AES256',
      Tagging: options.tags ? this.formatTags(options.tags) : undefined
    };

    if (options.multipart && this.shouldUseMultipart(scanData)) {
      return this.executeMultipartUpload(uploadParams);
    }

    const result = await this.executeWithRetry(() => 
      this.s3Client.putObject(uploadParams).promise()
    );

    const url = this.s3Client.getSignedUrl('getObject', {
      Bucket: LIDAR_BUCKET,
      Key: key,
      Expires: options.expiresIn || 3600
    });

    return {
      key,
      url,
      etag: result.ETag,
      versionId: result.VersionId,
      metadata: uploadParams.Metadata
    };
  }

  public async getLidarScan(
    scanId: string,
    options: GetOptions = {}
  ): Promise<GetResult> {
    const params: AWS.S3.GetObjectRequest = {
      Bucket: LIDAR_BUCKET,
      Key: `scans/${scanId}`,
      Range: options.range ? `bytes=${options.range.start}-${options.range.end}` : undefined
    };

    if (options.ifModifiedSince) {
      params.IfModifiedSince = options.ifModifiedSince;
    }

    const result = await this.executeWithRetry(() =>
      this.s3Client.getObject(params).promise()
    );

    return {
      data: options.streaming ? result.Body as Readable : result.Body as Buffer,
      metadata: result.Metadata || {},
      contentType: result.ContentType,
      contentLength: result.ContentLength,
      lastModified: result.LastModified
    };
  }

  private async executeMultipartUpload(
    params: AWS.S3.PutObjectRequest
  ): Promise<UploadResult> {
    const multipartUpload = await this.s3Client.createMultipartUpload(params).promise();
    const uploadId = multipartUpload.UploadId;
    const parts: AWS.S3.CompletedPart[] = [];

    try {
      const buffer = params.Body as Buffer;
      let partNumber = 1;
      let position = 0;

      while (position < buffer.length) {
        const size = Math.min(MULTIPART_THRESHOLD, buffer.length - position);
        const partBuffer = buffer.slice(position, position + size);

        const uploadPartParams = {
          ...params,
          UploadId: uploadId,
          PartNumber: partNumber,
          Body: partBuffer
        };

        const partResult = await this.executeWithRetry(() =>
          this.s3Client.uploadPart(uploadPartParams).promise()
        );

        parts.push({
          PartNumber: partNumber,
          ETag: partResult.ETag
        });

        position += size;
        partNumber++;
      }

      const completeParams = {
        Bucket: params.Bucket,
        Key: params.Key,
        UploadId: uploadId,
        MultipartUpload: { Parts: parts }
      };

      const result = await this.s3Client.completeMultipartUpload(completeParams).promise();

      return {
        key: params.Key,
        url: this.s3Client.getSignedUrl('getObject', {
          Bucket: params.Bucket,
          Key: params.Key,
          Expires: 3600
        }),
        etag: result.ETag,
        versionId: result.VersionId,
        metadata: params.Metadata
      };
    } catch (error) {
      await this.s3Client.abortMultipartUpload({
        Bucket: params.Bucket,
        Key: params.Key,
        UploadId: uploadId
      }).promise();
      throw error;
    }
  }

  private shouldUseMultipart(data: Buffer | Readable): boolean {
    if (Buffer.isBuffer(data)) {
      return data.length > MULTIPART_THRESHOLD;
    }
    return true;
  }

  private formatTags(tags: Record<string, string>): string {
    return Object.entries(tags)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
      .join('&');
  }
}