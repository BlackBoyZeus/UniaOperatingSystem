import Redis from 'ioredis'; // v5.3.2
import { MetricsCollector } from '@opentelemetry/metrics'; // v1.x
import { RedisConfig } from '../../config/redis.config';
import { FleetState } from '../../core/fleet/FleetState';
import { GameState } from '../../core/game/GameState';

// Constants for cache management
const DEFAULT_TTL = 900; // 15 minutes in seconds
const STATE_KEY_PREFIX = 'tald:state:';
const FLEET_KEY_PREFIX = 'tald:fleet:';
const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 50;

/**
 * Redis-based state caching service for TALD UNIA platform
 * Manages temporary storage and retrieval of fleet and game states
 * with CRDT support, automatic TTL management, and high availability
 */
export class StateCache {
    private redisCluster: Redis.Cluster;
    private config: RedisConfig;
    private metrics: MetricsCollector;
    private retryCount: Map<string, number> = new Map();

    constructor(config: RedisConfig, metrics: MetricsCollector) {
        this.config = config;
        this.metrics = metrics;
        this.initializeRedisCluster();
    }

    /**
     * Initializes Redis cluster connection with error handling and monitoring
     */
    private async initializeRedisCluster(): Promise<void> {
        try {
            this.redisCluster = await this.config.createRedisCluster();

            this.redisCluster.on('error', (error: Error) => {
                console.error('Redis cluster error:', error);
                this.metrics.recordError('redis_cluster_error');
            });

            this.redisCluster.on('ready', () => {
                console.info('Redis cluster ready');
                this.metrics.recordMetric('redis_cluster_ready', 1);
            });

            // Start cluster health monitoring
            setInterval(() => this.checkClusterHealth(), 5000);

        } catch (error) {
            console.error('Failed to initialize Redis cluster:', error);
            this.metrics.recordError('redis_cluster_init_error');
            throw error;
        }
    }

    /**
     * Caches fleet state with automatic TTL and performance monitoring
     */
    public async setFleetState(fleetId: string, state: FleetState): Promise<void> {
        const startTime = Date.now();
        const key = `${FLEET_KEY_PREFIX}${fleetId}`;

        try {
            if (!state.validateState()) {
                throw new Error('Invalid fleet state');
            }

            const serializedState = JSON.stringify(state);
            await this.redisCluster.setex(key, DEFAULT_TTL, serializedState);

            this.metrics.recordLatency('fleet_state_cache_write', Date.now() - startTime);
            this.metrics.recordMetric('fleet_state_cache_success', 1);

        } catch (error) {
            this.metrics.recordError('fleet_state_cache_error');
            this.handleRetry('setFleetState', fleetId, () => this.setFleetState(fleetId, state));
            throw error;
        }
    }

    /**
     * Retrieves cached fleet state with performance monitoring
     */
    public async getFleetState(fleetId: string): Promise<FleetState | null> {
        const startTime = Date.now();
        const key = `${FLEET_KEY_PREFIX}${fleetId}`;

        try {
            const cachedState = await this.redisCluster.get(key);
            
            if (!cachedState) {
                this.metrics.recordMetric('fleet_state_cache_miss', 1);
                return null;
            }

            const state = JSON.parse(cachedState);
            if (!FleetState.validateState(state)) {
                throw new Error('Invalid cached fleet state');
            }

            this.metrics.recordLatency('fleet_state_cache_read', Date.now() - startTime);
            this.metrics.recordMetric('fleet_state_cache_hit', 1);

            return state;

        } catch (error) {
            this.metrics.recordError('fleet_state_cache_read_error');
            this.handleRetry('getFleetState', fleetId, () => this.getFleetState(fleetId));
            throw error;
        }
    }

    /**
     * Caches game state with automatic TTL and performance monitoring
     */
    public async setGameState(gameId: string, state: GameState): Promise<void> {
        const startTime = Date.now();
        const key = `${STATE_KEY_PREFIX}${gameId}`;

        try {
            if (!state.validateState()) {
                throw new Error('Invalid game state');
            }

            const serializedState = JSON.stringify(state);
            await this.redisCluster.setex(key, DEFAULT_TTL, serializedState);

            this.metrics.recordLatency('game_state_cache_write', Date.now() - startTime);
            this.metrics.recordMetric('game_state_cache_success', 1);

        } catch (error) {
            this.metrics.recordError('game_state_cache_error');
            this.handleRetry('setGameState', gameId, () => this.setGameState(gameId, state));
            throw error;
        }
    }

    /**
     * Retrieves cached game state with performance monitoring
     */
    public async getGameState(gameId: string): Promise<GameState | null> {
        const startTime = Date.now();
        const key = `${STATE_KEY_PREFIX}${gameId}`;

        try {
            const cachedState = await this.redisCluster.get(key);
            
            if (!cachedState) {
                this.metrics.recordMetric('game_state_cache_miss', 1);
                return null;
            }

            const state = JSON.parse(cachedState);
            if (!GameState.validateState(state)) {
                throw new Error('Invalid cached game state');
            }

            this.metrics.recordLatency('game_state_cache_read', Date.now() - startTime);
            this.metrics.recordMetric('game_state_cache_hit', 1);

            return state;

        } catch (error) {
            this.metrics.recordError('game_state_cache_read_error');
            this.handleRetry('getGameState', gameId, () => this.getGameState(gameId));
            throw error;
        }
    }

    /**
     * Removes cached state for fleet or game with error handling
     */
    public async clearState(id: string, type: 'fleet' | 'game'): Promise<void> {
        const key = type === 'fleet' ? `${FLEET_KEY_PREFIX}${id}` : `${STATE_KEY_PREFIX}${id}`;

        try {
            await this.redisCluster.del(key);
            this.metrics.recordMetric(`${type}_state_cache_clear`, 1);

        } catch (error) {
            this.metrics.recordError(`${type}_state_cache_clear_error`);
            this.handleRetry('clearState', id, () => this.clearState(id, type));
            throw error;
        }
    }

    /**
     * Monitors Redis cluster health and triggers recovery if needed
     */
    private async checkClusterHealth(): Promise<boolean> {
        try {
            const health = await this.config.getClusterHealth();
            
            this.metrics.recordMetric('redis_cluster_nodes', health.nodesStatus.size);
            this.metrics.recordMetric('redis_cluster_health', 
                health.clusterState === 'ok' ? 1 : 0);

            // Monitor latencies
            for (const [node, latency] of health.latencies) {
                if (latency > 50) { // Alert if latency exceeds 50ms requirement
                    console.warn(`High latency detected for node ${node}: ${latency}ms`);
                    this.metrics.recordMetric('redis_high_latency_node', 1);
                }
            }

            return health.clusterState === 'ok';

        } catch (error) {
            console.error('Cluster health check failed:', error);
            this.metrics.recordError('redis_cluster_health_check_error');
            return false;
        }
    }

    /**
     * Handles operation retry with exponential backoff
     */
    private async handleRetry(
        operation: string, 
        id: string, 
        retryFn: () => Promise<any>
    ): Promise<void> {
        const retryCount = (this.retryCount.get(id) || 0) + 1;
        this.retryCount.set(id, retryCount);

        if (retryCount <= MAX_RETRY_ATTEMPTS) {
            const delay = RETRY_DELAY_MS * Math.pow(2, retryCount - 1);
            console.warn(`Retrying ${operation} for ${id}, attempt ${retryCount}`);
            
            await new Promise(resolve => setTimeout(resolve, delay));
            await retryFn();
            
            this.retryCount.delete(id);
        } else {
            this.retryCount.delete(id);
            throw new Error(`${operation} failed after ${MAX_RETRY_ATTEMPTS} attempts`);
        }
    }
}

export default StateCache;