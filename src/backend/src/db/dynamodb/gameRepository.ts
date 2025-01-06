import { DynamoDB, CloudWatch } from 'aws-sdk';
import * as Automerge from 'automerge';
import * as winston from 'winston';
import { injectable } from 'inversify';
import { IGameState, IEnvironmentState, IPhysicsState } from '../../interfaces/game.interface';
import { 
    GameCRDTDocument, 
    CRDTOperation, 
    SyncError, 
    SyncStats,
    DEFAULT_SYNC_INTERVAL,
    MAX_RETRIES,
    SYNC_TIMEOUT,
    BackoffStrategy
} from '../../types/crdt.types';

// Global constants
const GAME_TABLE_NAME = process.env.GAME_TABLE_NAME || 'tald-game-states';
const STATE_TTL_DAYS = 7;
const BATCH_SIZE = 25;
const MAX_RETRIES = 3;
const OPERATION_TIMEOUT_MS = 5000;

interface RetryConfig {
    maxRetries: number;
    backoffStrategy: BackoffStrategy;
    timeout: number;
}

@injectable()
export class GameRepository {
    private readonly crdt: Automerge.Doc<GameCRDTDocument>;
    private readonly syncStats: SyncStats;
    private readonly metrics: CloudWatch.MetricData[];

    constructor(
        private readonly dynamoClient: DynamoDB.DocumentClient,
        private readonly cloudWatch: CloudWatch,
        private readonly logger: winston.Logger
    ) {
        this.crdt = Automerge.init();
        this.syncStats = this.initializeSyncStats();
        this.metrics = [];
        this.setupCloudWatchMetrics();
    }

    private initializeSyncStats(): SyncStats {
        return {
            totalOperations: 0,
            successfulOperations: 0,
            failedOperations: 0,
            averageLatency: 0,
            lastSyncTimestamp: Date.now()
        };
    }

    private setupCloudWatchMetrics(): void {
        this.metrics.push({
            MetricName: 'GameStateOperationLatency',
            Unit: 'Milliseconds',
            Dimensions: [{ Name: 'Environment', Value: process.env.NODE_ENV || 'development' }]
        });
    }

    private async validateGameState(gameState: IGameState): Promise<boolean> {
        const startTime = Date.now();
        try {
            if (!gameState.gameId || !gameState.environment || !gameState.physics) {
                throw new Error('Invalid game state: missing required fields');
            }

            // Validate environment state
            if (!this.validateEnvironmentState(gameState.environment)) {
                throw new Error('Invalid environment state');
            }

            // Validate physics state
            if (!this.validatePhysicsState(gameState.physics)) {
                throw new Error('Invalid physics state');
            }

            // Validate CRDT compatibility
            const crdtDoc = Automerge.from(gameState);
            if (!crdtDoc) {
                throw new Error('Failed to create CRDT document');
            }

            return true;
        } catch (error) {
            this.logger.error('Game state validation failed', {
                error,
                gameId: gameState.gameId,
                duration: Date.now() - startTime
            });
            return false;
        }
    }

    private validateEnvironmentState(state: IEnvironmentState): boolean {
        return !!(
            state.timestamp &&
            typeof state.scanQuality === 'number' &&
            state.scanQuality >= 0 &&
            state.scanQuality <= 1 &&
            Array.isArray(state.classifiedObjects)
        );
    }

    private validatePhysicsState(state: IPhysicsState): boolean {
        return !!(
            state.timestamp &&
            Array.isArray(state.objects) &&
            Array.isArray(state.collisions) &&
            typeof state.simulationLatency === 'number'
        );
    }

    private async retryOperation<T>(
        operation: () => Promise<T>,
        config: RetryConfig
    ): Promise<T> {
        let retryCount = 0;
        let lastError: Error;

        while (retryCount < config.maxRetries) {
            try {
                const result = await Promise.race([
                    operation(),
                    new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Operation timeout')), config.timeout)
                    )
                ]);
                return result as T;
            } catch (error) {
                lastError = error as Error;
                retryCount++;
                
                if (retryCount < config.maxRetries) {
                    const delay = this.calculateBackoff(retryCount, config.backoffStrategy);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw lastError;
    }

    private calculateBackoff(attempt: number, strategy: BackoffStrategy): number {
        switch (strategy) {
            case BackoffStrategy.EXPONENTIAL:
                return Math.min(1000 * Math.pow(2, attempt), 10000);
            case BackoffStrategy.FIBONACCI:
                return Math.min(this.getFibonacci(attempt) * 1000, 10000);
            default:
                return 1000 * attempt;
        }
    }

    private getFibonacci(n: number): number {
        if (n <= 1) return n;
        let prev = 0, curr = 1;
        for (let i = 2; i <= n; i++) {
            const next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }

    public async createGameState(gameState: IGameState): Promise<IGameState> {
        const startTime = Date.now();
        try {
            // Validate input state
            if (!await this.validateGameState(gameState)) {
                throw new Error('Game state validation failed');
            }

            // Initialize CRDT document
            const crdtDoc = Automerge.change(this.crdt, doc => {
                doc.data = gameState;
                doc.version = 1;
                doc.lastSyncTimestamp = Date.now();
            });

            // Prepare DynamoDB item
            const item = {
                gameId: gameState.gameId,
                sessionId: gameState.sessionId,
                fleetId: gameState.fleetId,
                state: Automerge.save(crdtDoc),
                ttl: Math.floor(Date.now() / 1000) + (STATE_TTL_DAYS * 86400),
                createdAt: Date.now(),
                updatedAt: Date.now()
            };

            // Persist to DynamoDB with retry logic
            await this.retryOperation(
                () => this.dynamoClient.put({
                    TableName: GAME_TABLE_NAME,
                    Item: item,
                    ConditionExpression: 'attribute_not_exists(gameId)'
                }).promise(),
                {
                    maxRetries: MAX_RETRIES,
                    backoffStrategy: BackoffStrategy.EXPONENTIAL,
                    timeout: OPERATION_TIMEOUT_MS
                }
            );

            // Update metrics
            this.updateOperationMetrics('createGameState', startTime);

            return gameState;
        } catch (error) {
            this.handleOperationError('createGameState', error, gameState.gameId);
            throw error;
        }
    }

    public async updateGameState(gameState: IGameState): Promise<IGameState> {
        const startTime = Date.now();
        try {
            // Validate input state
            if (!await this.validateGameState(gameState)) {
                throw new Error('Game state validation failed');
            }

            // Get existing state
            const existingItem = await this.getGameState(gameState.gameId);
            if (!existingItem) {
                throw new Error('Game state not found');
            }

            // Merge CRDT states
            const existingDoc = Automerge.load(existingItem.state);
            const newDoc = Automerge.change(existingDoc, doc => {
                doc.data = gameState;
                doc.version += 1;
                doc.lastSyncTimestamp = Date.now();
            });

            // Prepare update item
            const updateItem = {
                gameId: gameState.gameId,
                state: Automerge.save(newDoc),
                updatedAt: Date.now()
            };

            // Update DynamoDB with retry logic
            await this.retryOperation(
                () => this.dynamoClient.update({
                    TableName: GAME_TABLE_NAME,
                    Key: { gameId: gameState.gameId },
                    UpdateExpression: 'SET #state = :state, updatedAt = :updatedAt',
                    ExpressionAttributeNames: { '#state': 'state' },
                    ExpressionAttributeValues: {
                        ':state': updateItem.state,
                        ':updatedAt': updateItem.updatedAt
                    }
                }).promise(),
                {
                    maxRetries: MAX_RETRIES,
                    backoffStrategy: BackoffStrategy.EXPONENTIAL,
                    timeout: OPERATION_TIMEOUT_MS
                }
            );

            // Update metrics
            this.updateOperationMetrics('updateGameState', startTime);

            return gameState;
        } catch (error) {
            this.handleOperationError('updateGameState', error, gameState.gameId);
            throw error;
        }
    }

    public async getGameState(gameId: string): Promise<IGameState | null> {
        const startTime = Date.now();
        try {
            const result = await this.retryOperation(
                () => this.dynamoClient.get({
                    TableName: GAME_TABLE_NAME,
                    Key: { gameId }
                }).promise(),
                {
                    maxRetries: MAX_RETRIES,
                    backoffStrategy: BackoffStrategy.EXPONENTIAL,
                    timeout: OPERATION_TIMEOUT_MS
                }
            );

            if (!result.Item) {
                return null;
            }

            const crdtDoc = Automerge.load(result.Item.state);
            this.updateOperationMetrics('getGameState', startTime);

            return crdtDoc.data;
        } catch (error) {
            this.handleOperationError('getGameState', error, gameId);
            throw error;
        }
    }

    public async deleteGameState(gameId: string): Promise<void> {
        const startTime = Date.now();
        try {
            await this.retryOperation(
                () => this.dynamoClient.delete({
                    TableName: GAME_TABLE_NAME,
                    Key: { gameId }
                }).promise(),
                {
                    maxRetries: MAX_RETRIES,
                    backoffStrategy: BackoffStrategy.EXPONENTIAL,
                    timeout: OPERATION_TIMEOUT_MS
                }
            );

            this.updateOperationMetrics('deleteGameState', startTime);
        } catch (error) {
            this.handleOperationError('deleteGameState', error, gameId);
            throw error;
        }
    }

    private updateOperationMetrics(operation: string, startTime: number): void {
        const duration = Date.now() - startTime;
        this.syncStats.totalOperations++;
        this.syncStats.successfulOperations++;
        this.syncStats.averageLatency = 
            (this.syncStats.averageLatency * (this.syncStats.totalOperations - 1) + duration) / 
            this.syncStats.totalOperations;
        this.syncStats.lastSyncTimestamp = Date.now();

        // Push metrics to CloudWatch
        this.cloudWatch.putMetricData({
            Namespace: 'TALD/GameRepository',
            MetricData: [{
                ...this.metrics[0],
                Value: duration,
                Timestamp: new Date(),
                Dimensions: [
                    ...this.metrics[0].Dimensions!,
                    { Name: 'Operation', Value: operation }
                ]
            }]
        }).promise().catch(error => {
            this.logger.error('Failed to push metrics to CloudWatch', { error });
        });
    }

    private handleOperationError(operation: string, error: any, gameId: string): void {
        this.syncStats.failedOperations++;
        this.logger.error(`${operation} failed`, {
            error,
            gameId,
            syncStats: this.syncStats
        });

        // Push error metrics to CloudWatch
        this.cloudWatch.putMetricData({
            Namespace: 'TALD/GameRepository/Errors',
            MetricData: [{
                MetricName: 'OperationErrors',
                Value: 1,
                Unit: 'Count',
                Timestamp: new Date(),
                Dimensions: [
                    { Name: 'Operation', Value: operation },
                    { Name: 'ErrorType', Value: error.name || 'Unknown' }
                ]
            }]
        }).promise().catch(error => {
            this.logger.error('Failed to push error metrics to CloudWatch', { error });
        });
    }
}