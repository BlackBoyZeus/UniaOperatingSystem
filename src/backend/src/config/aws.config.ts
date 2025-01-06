import AWS from 'aws-sdk'; // v2.1450.0
import dotenv from 'dotenv'; // v16.3.1
import { TaldLogger } from '../utils/logger.utils';

// Load environment variables
dotenv.config();

// Global configuration constants
const AWS_PRIMARY_REGION = process.env.AWS_PRIMARY_REGION || 'us-east-1';
const AWS_SECONDARY_REGIONS = process.env.AWS_SECONDARY_REGIONS?.split(',') || ['us-west-2', 'eu-west-1'];
const DYNAMODB_ENDPOINTS = process.env.DYNAMODB_ENDPOINTS;
const S3_BUCKET_REGIONS = process.env.S3_BUCKET_REGIONS?.split(',') || ['us-east-1', 'us-west-2'];
const CLOUDWATCH_REGIONS = process.env.CLOUDWATCH_REGIONS?.split(',') || ['us-east-1'];

// Service-specific configuration constants
const DYNAMODB_CONFIG = {
  maxRetries: 3,
  timeout: 5000,
  enableAutoScaling: true,
  backupEnabled: true,
  pointInTimeRecovery: true,
  encryptionAtRest: true,
};

const S3_CONFIG = {
  versioning: true,
  encryption: 'AES256',
  replicationEnabled: true,
  lifecycleDays: 365,
  serverAccessLogging: true,
};

const CLOUDWATCH_CONFIG = {
  retentionDays: 30,
  metricResolution: 60,
  alarmEvaluationPeriods: 3,
  anomalyDetectionEnabled: true,
};

/**
 * Validates AWS configuration settings including credentials, endpoints, and region settings
 */
export const validateConfig = (config: any): boolean => {
  try {
    // Verify AWS credentials
    if (!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY) {
      throw new Error('AWS credentials not found');
    }

    // Validate regions
    const allRegions = [AWS_PRIMARY_REGION, ...AWS_SECONDARY_REGIONS];
    for (const region of allRegions) {
      if (!AWS.config.regionRegex.test(region)) {
        throw new Error(`Invalid AWS region: ${region}`);
      }
    }

    // Validate endpoints if provided
    if (DYNAMODB_ENDPOINTS) {
      const endpoints = DYNAMODB_ENDPOINTS.split(',');
      for (const endpoint of endpoints) {
        new URL(endpoint);
      }
    }

    // Validate service-specific configurations
    if (!config.dynamoDBClients || !config.s3Clients || !config.cloudWatchClients) {
      throw new Error('Required service clients not initialized');
    }

    return true;
  } catch (error) {
    throw new Error(`Configuration validation failed: ${error.message}`);
  }
};

/**
 * Enhanced configuration class for AWS services with multi-region support
 */
export class AWSConfig {
  private dynamoDBClients: AWS.DynamoDB.DocumentClient[];
  private s3Clients: AWS.S3[];
  private cloudWatchClients: AWS.CloudWatch[];
  private logger: TaldLogger;

  constructor() {
    this.logger = new TaldLogger({
      serviceName: 'aws-config',
      environment: process.env.NODE_ENV || 'development',
      enableCloudWatch: true,
      performanceTracking: true,
      securitySettings: {
        trackAuthEvents: true,
        trackSystemIntegrity: true,
        fleetTrustThreshold: 80,
      },
    });

    // Configure AWS SDK defaults
    AWS.config.update({
      maxRetries: 3,
      httpOptions: { timeout: 5000 },
      logger: console,
    });

    this.initializeClients();
  }

  private initializeClients(): void {
    try {
      // Initialize DynamoDB clients for each region
      this.dynamoDBClients = [AWS_PRIMARY_REGION, ...AWS_SECONDARY_REGIONS].map(region => {
        const config = { region };
        if (DYNAMODB_ENDPOINTS) {
          config['endpoint'] = DYNAMODB_ENDPOINTS.split(',')[0];
        }
        return new AWS.DynamoDB.DocumentClient(config);
      });

      // Initialize S3 clients for specified regions
      this.s3Clients = S3_BUCKET_REGIONS.map(region => 
        new AWS.S3({ region })
      );

      // Initialize CloudWatch clients
      this.cloudWatchClients = CLOUDWATCH_REGIONS.map(region =>
        new AWS.CloudWatch({ region })
      );

      this.logger.info('AWS clients initialized successfully', {
        regions: {
          primary: AWS_PRIMARY_REGION,
          secondary: AWS_SECONDARY_REGIONS,
        },
      });
    } catch (error) {
      this.logger.error('Failed to initialize AWS clients', error);
      throw error;
    }
  }

  /**
   * Returns enhanced DynamoDB configuration with multi-region support
   */
  public getDynamoDBConfig(): object {
    return {
      clients: this.dynamoDBClients,
      primary: this.dynamoDBClients[0],
      config: {
        ...DYNAMODB_CONFIG,
        regions: [AWS_PRIMARY_REGION, ...AWS_SECONDARY_REGIONS],
        tableOptions: {
          BillingMode: 'PAY_PER_REQUEST',
          StreamSpecification: {
            StreamEnabled: true,
            StreamViewType: 'NEW_AND_OLD_IMAGES',
          },
        },
        backupConfig: {
          enabled: DYNAMODB_CONFIG.backupEnabled,
          frequency: 'daily',
          retention: 30,
        },
      },
    };
  }

  /**
   * Returns enhanced S3 configuration with replication and lifecycle policies
   */
  public getS3Config(): object {
    return {
      clients: this.s3Clients,
      primary: this.s3Clients[0],
      config: {
        ...S3_CONFIG,
        regions: S3_BUCKET_REGIONS,
        bucketConfig: {
          CORSConfiguration: {
            CORSRules: [{
              AllowedHeaders: ['*'],
              AllowedMethods: ['GET', 'PUT', 'POST', 'DELETE'],
              AllowedOrigins: ['*'],
              MaxAgeSeconds: 3000,
            }],
          },
          VersioningConfiguration: {
            Status: 'Enabled',
          },
          ReplicationConfiguration: {
            Role: process.env.AWS_REPLICATION_ROLE_ARN,
            Rules: [{
              Status: 'Enabled',
              Priority: 1,
              DeleteMarkerReplication: { Status: 'Enabled' },
              Destination: {
                Bucket: process.env.AWS_REPLICATION_BUCKET_ARN,
                Account: process.env.AWS_ACCOUNT_ID,
              },
            }],
          },
        },
      },
    };
  }

  /**
   * Returns enhanced CloudWatch configuration with comprehensive monitoring
   */
  public getCloudWatchConfig(): object {
    return {
      clients: this.cloudWatchClients,
      primary: this.cloudWatchClients[0],
      config: {
        ...CLOUDWATCH_CONFIG,
        regions: CLOUDWATCH_REGIONS,
        logGroupConfig: {
          retentionInDays: CLOUDWATCH_CONFIG.retentionDays,
          encryptionEnabled: true,
        },
        metricsConfig: {
          namespace: 'TALD/UNIA',
          dimensions: ['Region', 'Service', 'Operation'],
          resolution: CLOUDWATCH_CONFIG.metricResolution,
        },
        alarmConfig: {
          evaluationPeriods: CLOUDWATCH_CONFIG.alarmEvaluationPeriods,
          datapointsToAlarm: 2,
          treatMissingData: 'breaching',
        },
        dashboards: {
          fleetHealth: {
            widgets: ['NetworkLatency', 'DeviceCount', 'ErrorRate'],
            refreshInterval: 60,
          },
          systemMetrics: {
            widgets: ['CPU', 'Memory', 'DiskIO'],
            refreshInterval: 300,
          },
        },
      },
    };
  }
}