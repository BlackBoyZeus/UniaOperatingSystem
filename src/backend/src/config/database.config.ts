import dotenv from 'dotenv'; // v16.3.1
import { AWSConfig } from './aws.config';
import { RedisConfig } from './redis.config';
import { TaldLogger } from '../utils/logger.utils';

// Load environment variables
dotenv.config();

// Database configuration constants
const DB_REGION = process.env.DB_REGION || 'us-east-1';
const DB_AUTO_SCALE = process.env.DB_AUTO_SCALE === 'true';
const DB_MIN_CAPACITY = process.env.DB_MIN_CAPACITY || 5;
const DB_MAX_CAPACITY = process.env.DB_MAX_CAPACITY || 100;
const DB_BACKUP_ENABLED = process.env.DB_BACKUP_ENABLED === 'true';
const DB_ENCRYPTION_KEY = process.env.DB_ENCRYPTION_KEY;
const DB_COMPLIANCE_MODE = process.env.DB_COMPLIANCE_MODE || 'standard';

// Performance thresholds
const PERFORMANCE_THRESHOLDS = {
  maxLatency: 50, // ms
  minThroughput: 1000, // requests/second
  maxConnectionPool: 100,
  maxMemoryUsage: 0.85, // 85% threshold
};

/**
 * Enhanced configuration class for database services with security and performance features
 */
export class DatabaseConfig {
  private awsConfig: AWSConfig;
  private redisConfig: RedisConfig;
  private logger: TaldLogger;
  private performanceMetrics: Map<string, number>;
  private securityConfig: {
    encryptionEnabled: boolean;
    auditLogging: boolean;
    complianceMode: string;
  };

  constructor() {
    this.logger = new TaldLogger({
      serviceName: 'database-config',
      environment: process.env.NODE_ENV || 'development',
      enableCloudWatch: true,
      performanceTracking: true,
      securitySettings: {
        trackAuthEvents: true,
        trackSystemIntegrity: true,
        fleetTrustThreshold: 80,
      },
    });

    this.awsConfig = new AWSConfig();
    this.redisConfig = new RedisConfig();
    this.performanceMetrics = new Map();
    this.securityConfig = {
      encryptionEnabled: !!DB_ENCRYPTION_KEY,
      auditLogging: true,
      complianceMode: DB_COMPLIANCE_MODE,
    };
  }

  /**
   * Initializes all database connections with security and performance monitoring
   */
  public async initialize(): Promise<void> {
    try {
      // Validate configuration before initialization
      if (!validateDatabaseConfig(this)) {
        throw new Error('Invalid database configuration');
      }

      // Initialize DynamoDB configuration
      const dynamoConfig = this.awsConfig.getDynamoDBConfig();
      this.logger.info('DynamoDB configuration initialized', { region: DB_REGION });

      // Initialize Redis cluster
      await this.redisConfig.createRedisCluster();
      this.logger.info('Redis cluster initialized');

      // Set up performance monitoring
      this.initializePerformanceMonitoring();

      // Configure security settings
      this.initializeSecuritySettings();

      this.logger.info('Database configuration initialized successfully', {
        region: DB_REGION,
        autoScale: DB_AUTO_SCALE,
        securityEnabled: this.securityConfig.encryptionEnabled,
      });
    } catch (error) {
      this.logger.error('Failed to initialize database configuration', error);
      throw error;
    }
  }

  /**
   * Returns enhanced database auto-scaling configuration with performance optimization
   */
  public getScalingConfig(): object {
    return {
      enabled: DB_AUTO_SCALE,
      minCapacity: parseInt(DB_MIN_CAPACITY.toString(), 10),
      maxCapacity: parseInt(DB_MAX_CAPACITY.toString(), 10),
      targetUtilization: 70,
      scaleInCooldown: 60,
      scaleOutCooldown: 60,
      performanceMonitoring: {
        enabled: true,
        metrics: Array.from(this.performanceMetrics.entries()),
        thresholds: PERFORMANCE_THRESHOLDS,
      },
    };
  }

  private initializePerformanceMonitoring(): void {
    // Set up performance metric collection
    this.performanceMetrics.set('latency', 0);
    this.performanceMetrics.set('throughput', 0);
    this.performanceMetrics.set('errorRate', 0);
    this.performanceMetrics.set('connectionCount', 0);

    // Schedule periodic performance checks
    setInterval(async () => {
      try {
        const health = await getDatabaseHealth();
        this.updatePerformanceMetrics(health);
      } catch (error) {
        this.logger.error('Performance monitoring error', error);
      }
    }, 60000); // Check every minute
  }

  private initializeSecuritySettings(): void {
    if (this.securityConfig.encryptionEnabled) {
      this.logger.info('Database encryption enabled', {
        mode: this.securityConfig.complianceMode,
      });
    }

    if (this.securityConfig.auditLogging) {
      this.logger.info('Audit logging enabled for database operations');
    }
  }

  private updatePerformanceMetrics(health: any): void {
    this.performanceMetrics.set('latency', health.latency);
    this.performanceMetrics.set('throughput', health.throughput);
    this.performanceMetrics.set('errorRate', health.errorRate);
    this.performanceMetrics.set('connectionCount', health.connections);

    // Log performance alerts if thresholds are exceeded
    if (health.latency > PERFORMANCE_THRESHOLDS.maxLatency) {
      this.logger.warn('High database latency detected', {
        current: health.latency,
        threshold: PERFORMANCE_THRESHOLDS.maxLatency,
      });
    }
  }
}

/**
 * Validates database configuration settings with enhanced security and compliance checks
 */
export function validateDatabaseConfig(config: any): boolean {
  try {
    // Verify required environment variables
    if (!DB_REGION || !DB_ENCRYPTION_KEY) {
      throw new Error('Missing required database configuration');
    }

    // Validate scaling parameters
    if (DB_AUTO_SCALE) {
      const minCapacity = parseInt(DB_MIN_CAPACITY.toString(), 10);
      const maxCapacity = parseInt(DB_MAX_CAPACITY.toString(), 10);
      if (minCapacity >= maxCapacity) {
        throw new Error('Invalid scaling capacity configuration');
      }
    }

    // Verify security settings
    if (!config.securityConfig.encryptionEnabled) {
      throw new Error('Database encryption must be enabled');
    }

    return true;
  } catch (error) {
    throw new Error(`Database configuration validation failed: ${error.message}`);
  }
}

/**
 * Returns comprehensive health status of all database services with performance metrics
 */
export async function getDatabaseHealth(): Promise<object> {
  const redisConfig = new RedisConfig();
  const health = {
    timestamp: new Date().toISOString(),
    status: 'healthy',
    latency: 0,
    throughput: 0,
    errorRate: 0,
    connections: 0,
    services: {
      dynamodb: await checkDynamoDBHealth(),
      redis: await redisConfig.getClusterHealth(),
    },
  };

  // Aggregate performance metrics
  health.latency = Math.max(
    health.services.dynamodb.latency,
    health.services.redis.latency
  );
  health.connections = 
    health.services.dynamodb.connections +
    health.services.redis.connections;

  // Update overall status if any service is degraded
  if (health.services.dynamodb.status !== 'healthy' ||
      health.services.redis.clusterState !== 'ok') {
    health.status = 'degraded';
  }

  return health;
}

async function checkDynamoDBHealth(): Promise<object> {
  try {
    const awsConfig = new AWSConfig();
    const dynamoConfig = awsConfig.getDynamoDBConfig();
    
    return {
      status: 'healthy',
      latency: 0, // Implement actual latency check
      connections: 0, // Implement connection count
      region: DB_REGION,
      autoScaling: DB_AUTO_SCALE,
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error.message,
    };
  }
}