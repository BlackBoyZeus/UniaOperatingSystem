import Redis from 'ioredis'; // v5.3.2
import dotenv from 'dotenv'; // v16.3.1

dotenv.config();

// Redis cluster node configuration
const REDIS_CLUSTER_NODES = process.env.REDIS_NODES?.split(',') || ['localhost:6379'];

// Redis connection options with security and high availability settings
const REDIS_OPTIONS: Redis.ClusterOptions = {
  password: process.env.REDIS_PASSWORD,
  tls: process.env.REDIS_TLS === 'true',
  maxRetriesPerRequest: 3,
  retryStrategy: (times: number) => Math.min(times * 50, 2000),
  enableReadyCheck: true,
  scaleReads: 'all',
  maxRedirections: 16,
  retryDelayOnFailover: 100
};

// Default TTL for session data (15 minutes in seconds)
const DEFAULT_TTL_SECONDS = 900;

// Interface for cluster health metrics
interface ClusterHealth {
  nodesStatus: Map<string, boolean>;
  latencies: Map<string, number>;
  memoryUsage: Map<string, number>;
  replicationStatus: boolean;
  clusterState: string;
}

export class RedisConfig {
  private redisCluster: Redis.Cluster | null = null;
  private isInitialized: boolean = false;
  private nodeLatencies: Map<string, number> = new Map();

  constructor() {
    this.validateEnvironment();
  }

  private validateEnvironment(): void {
    if (!process.env.REDIS_PASSWORD) {
      throw new Error('Redis password must be configured in environment variables');
    }
  }

  /**
   * Creates and initializes a Redis cluster connection with automatic failover support
   * @returns Promise<Redis.Cluster> Initialized Redis cluster instance
   */
  public async createRedisCluster(): Promise<Redis.Cluster> {
    try {
      this.redisCluster = new Redis.Cluster(REDIS_CLUSTER_NODES, {
        ...REDIS_OPTIONS,
        clusterRetryStrategy: (times: number) => {
          const delay = Math.min(times * 100, 2000);
          console.warn(`Cluster connection attempt ${times}, retrying in ${delay}ms`);
          return delay;
        }
      });

      // Set up event listeners for cluster management
      this.redisCluster.on('connect', () => {
        console.info('Connected to Redis cluster');
        this.isInitialized = true;
      });

      this.redisCluster.on('node error', (error: Error, node: string) => {
        console.error(`Redis node ${node} error:`, error);
        this.nodeLatencies.set(node, Infinity);
      });

      this.redisCluster.on('ready', () => {
        console.info('Redis cluster is ready');
        this.monitorLatency();
      });

      await this.waitForConnection();
      return this.redisCluster;

    } catch (error) {
      console.error('Failed to create Redis cluster:', error);
      throw error;
    }
  }

  /**
   * Comprehensive health check of Redis cluster including node status and latency
   * @returns Promise<ClusterHealth> Detailed cluster health status
   */
  public async getClusterHealth(): Promise<ClusterHealth> {
    if (!this.redisCluster || !this.isInitialized) {
      throw new Error('Redis cluster not initialized');
    }

    const nodes = this.redisCluster.nodes('master');
    const nodesStatus = new Map<string, boolean>();
    const memoryUsage = new Map<string, number>();

    for (const node of nodes) {
      try {
        const info = await node.info();
        nodesStatus.set(node.options.key, node.status === 'ready');
        memoryUsage.set(node.options.key, this.parseMemoryUsage(info));
      } catch (error) {
        console.error(`Failed to get node ${node.options.key} health:`, error);
        nodesStatus.set(node.options.key, false);
      }
    }

    return {
      nodesStatus,
      latencies: this.nodeLatencies,
      memoryUsage,
      replicationStatus: await this.checkReplicationStatus(),
      clusterState: await this.getClusterState()
    };
  }

  /**
   * Safely disconnects from Redis cluster ensuring data persistence
   * @returns Promise<void>
   */
  public async disconnect(): Promise<void> {
    if (this.redisCluster && this.isInitialized) {
      try {
        await this.redisCluster.quit();
        this.isInitialized = false;
        this.nodeLatencies.clear();
        console.info('Disconnected from Redis cluster');
      } catch (error) {
        console.error('Error disconnecting from Redis cluster:', error);
        throw error;
      }
    }
  }

  /**
   * Monitors and tracks latency for each cluster node
   * @returns Promise<Map<string, number>> Node latency measurements
   */
  public async monitorLatency(): Promise<Map<string, number>> {
    if (!this.redisCluster || !this.isInitialized) {
      throw new Error('Redis cluster not initialized');
    }

    const nodes = this.redisCluster.nodes();
    for (const node of nodes) {
      try {
        const start = Date.now();
        await node.ping();
        const latency = Date.now() - start;
        this.nodeLatencies.set(node.options.key, latency);

        if (latency > 50) { // Alert if latency exceeds 50ms requirement
          console.warn(`High latency detected for node ${node.options.key}: ${latency}ms`);
        }
      } catch (error) {
        console.error(`Failed to measure latency for node ${node.options.key}:`, error);
        this.nodeLatencies.set(node.options.key, Infinity);
      }
    }

    return this.nodeLatencies;
  }

  private async waitForConnection(timeout: number = 5000): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('Redis cluster connection timeout'));
      }, timeout);

      this.redisCluster?.once('ready', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  private async checkReplicationStatus(): Promise<boolean> {
    if (!this.redisCluster) return false;

    try {
      const nodes = this.redisCluster.nodes('slave');
      for (const node of nodes) {
        const info = await node.info('replication');
        if (!info.includes('master_link_status:up')) {
          return false;
        }
      }
      return true;
    } catch (error) {
      console.error('Failed to check replication status:', error);
      return false;
    }
  }

  private async getClusterState(): Promise<string> {
    if (!this.redisCluster) return 'unavailable';

    try {
      const clusterInfo = await this.redisCluster.cluster('info');
      return clusterInfo.includes('cluster_state:ok') ? 'ok' : 'degraded';
    } catch (error) {
      console.error('Failed to get cluster state:', error);
      return 'error';
    }
  }

  private parseMemoryUsage(info: string): number {
    const match = info.match(/used_memory:(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  }
}

export default RedisConfig;