import Redis from 'ioredis'; // v5.3.2
import CircuitBreaker from 'opossum'; // v6.0.0
import { RedisConfig, createRedisCluster, getClusterHealth, getConnectionPool } from '../../config/redis.config';
import { 
    ISession, 
    ISessionState, 
    SessionStatus, 
    IPerformanceMetrics 
} from '../../interfaces/session.interface';

// Global constants for session management
const SESSION_PREFIX = 'session:';
const SESSION_TTL = 900; // 15 minutes in seconds
const MAX_RETRY_ATTEMPTS = 3;
const MAX_FLEET_SIZE = 32;
const LOCK_TIMEOUT = 50; // 50ms lock timeout for optimistic locking
const CIRCUIT_BREAKER_TIMEOUT = 1000; // 1 second timeout for circuit breaker

/**
 * Redis-based session cache implementation with high availability and performance monitoring
 */
@CircuitBreaker({
    timeout: CIRCUIT_BREAKER_TIMEOUT,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
})
export class SessionCache {
    private redisCluster: Redis.Cluster;
    private readonly config: RedisConfig;
    private readonly breaker: CircuitBreaker;
    private readonly performanceMetrics: Map<string, IPerformanceMetrics>;

    constructor(config: RedisConfig) {
        this.config = config;
        this.performanceMetrics = new Map();
        this.initializeCache();
    }

    /**
     * Initializes Redis cluster connection and circuit breaker
     */
    private async initializeCache(): Promise<void> {
        try {
            this.redisCluster = await this.config.createRedisCluster();
            
            this.breaker = new CircuitBreaker(async () => {
                const health = await this.config.getClusterHealth();
                return health.clusterState === 'ok';
            }, {
                timeout: CIRCUIT_BREAKER_TIMEOUT,
                errorThresholdPercentage: 50,
                resetTimeout: 30000
            });

            this.redisCluster.on('error', (error: Error) => {
                console.error('Redis cluster error:', error);
                this.breaker.fire();
            });

        } catch (error) {
            console.error('Failed to initialize session cache:', error);
            throw error;
        }
    }

    /**
     * Stores session data with automatic expiration and fleet size validation
     */
    public async setSession(sessionId: string, sessionData: ISession): Promise<void> {
        const startTime = Date.now();
        
        try {
            // Validate fleet size
            if (sessionData.participants.length > MAX_FLEET_SIZE) {
                throw new Error(`Fleet size exceeds maximum limit of ${MAX_FLEET_SIZE}`);
            }

            const key = `${SESSION_PREFIX}${sessionId}`;
            const serializedData = JSON.stringify({
                ...sessionData,
                lastUpdate: Date.now(),
                performance: this.getPerformanceMetrics(sessionId)
            });

            await this.breaker.fire(async () => {
                const result = await this.redisCluster.setex(key, SESSION_TTL, serializedData);
                if (result !== 'OK') {
                    throw new Error('Failed to store session data');
                }
            });

            // Update performance metrics
            this.updatePerformanceMetrics(sessionId, {
                operationLatency: Date.now() - startTime,
                operation: 'setSession'
            });

        } catch (error) {
            console.error(`Failed to set session ${sessionId}:`, error);
            throw error;
        }
    }

    /**
     * Updates session state with optimistic locking for real-time synchronization
     */
    public async updateSessionState(sessionId: string, state: ISessionState): Promise<boolean> {
        const lockKey = `${SESSION_PREFIX}${sessionId}:lock`;
        const startTime = Date.now();

        try {
            // Acquire lock with timeout
            const locked = await this.redisCluster.set(lockKey, '1', 'PX', LOCK_TIMEOUT, 'NX');
            if (!locked) {
                throw new Error('Failed to acquire lock for session update');
            }

            // Get current session
            const key = `${SESSION_PREFIX}${sessionId}`;
            const currentSession = await this.redisCluster.get(key);
            
            if (!currentSession) {
                throw new Error('Session not found');
            }

            const session: ISession = JSON.parse(currentSession);

            // Validate fleet size and state
            if (state.activeParticipants > MAX_FLEET_SIZE) {
                throw new Error(`Active participants exceed maximum fleet size of ${MAX_FLEET_SIZE}`);
            }

            // Update session state
            const updatedSession: ISession = {
                ...session,
                state: {
                    ...state,
                    lastUpdate: new Date(),
                    performanceMetrics: this.getPerformanceMetrics(sessionId)
                }
            };

            // Store updated session with TTL refresh
            await this.redisCluster.setex(key, SESSION_TTL, JSON.stringify(updatedSession));

            // Update performance metrics
            this.updatePerformanceMetrics(sessionId, {
                operationLatency: Date.now() - startTime,
                operation: 'updateState'
            });

            return true;

        } catch (error) {
            console.error(`Failed to update session state ${sessionId}:`, error);
            return false;

        } finally {
            // Release lock
            await this.redisCluster.del(lockKey);
        }
    }

    /**
     * Retrieves session performance metrics
     */
    public async getPerformanceMetrics(sessionId: string): Promise<IPerformanceMetrics> {
        const metrics = this.performanceMetrics.get(sessionId) || this.createDefaultMetrics();
        
        // Validate session existence
        const exists = await this.redisCluster.exists(`${SESSION_PREFIX}${sessionId}`);
        if (!exists) {
            throw new Error('Session not found');
        }

        return metrics;
    }

    /**
     * Updates performance metrics for session operations
     */
    private updatePerformanceMetrics(sessionId: string, data: { 
        operationLatency: number; 
        operation: string; 
    }): void {
        const current = this.performanceMetrics.get(sessionId) || this.createDefaultMetrics();
        const updated: IPerformanceMetrics = {
            ...current,
            averageLatency: this.calculateAverageLatency(current.averageLatency, data.operationLatency),
            lastUpdate: Date.now()
        };

        this.performanceMetrics.set(sessionId, updated);
    }

    /**
     * Creates default performance metrics
     */
    private createDefaultMetrics(): IPerformanceMetrics {
        return {
            averageLatency: 0,
            packetLoss: 0,
            syncRate: 0,
            participantMetrics: new Map(),
            cpuUsage: 0,
            memoryUsage: 0,
            batteryLevel: 100,
            networkBandwidth: 0,
            scanQuality: 0,
            frameRate: 0,
            lastUpdate: Date.now()
        };
    }

    /**
     * Calculates exponential moving average for latency
     */
    private calculateAverageLatency(current: number, new_value: number, alpha: number = 0.2): number {
        return alpha * new_value + (1 - alpha) * current;
    }
}

export default SessionCache;