import { injectable } from 'inversify';
import { WebSocket, WebSocketServer } from 'ws';  // v8.13.0
import { Logger } from 'winston';  // v3.10.0
import { Counter, Gauge, Histogram } from 'prometheus-client';  // v14.2.0
import * as zlib from 'zlib';  // v1.0.5
import CircuitBreaker from 'circuit-breaker-js';  // v0.0.1
import * as genericPool from 'generic-pool';  // v3.9.0

import { FleetWebSocketHandler } from './handlers/fleetHandler';
import { GameHandler } from './handlers/gameHandler';
import { SessionHandler } from './handlers/sessionHandler';

// Constants for WebSocket configuration
const HEARTBEAT_INTERVAL = 30000;
const CONNECTION_TIMEOUT = 45000;
const MAX_MESSAGE_SIZE = 1048576; // 1MB
const CIRCUIT_BREAKER_THRESHOLD = 5;
const RATE_LIMIT_WINDOW = 60000;
const RATE_LIMIT_MAX_REQUESTS = 1000;
const CONNECTION_POOL_SIZE = 100;
const RECONNECTION_ATTEMPTS = 3;
const HEALTH_CHECK_INTERVAL = 15000;

@injectable()
export class WebSocketManager {
    private wss: WebSocketServer;
    private readonly connections: Map<string, WebSocket>;
    private readonly messageCounter: Counter;
    private readonly connectionGauge: Gauge;
    private readonly latencyHistogram: Histogram;
    private readonly circuitBreaker: CircuitBreaker;
    private readonly connectionPool: genericPool.Pool<WebSocket>;

    constructor(
        private readonly fleetHandler: FleetWebSocketHandler,
        private readonly gameHandler: GameHandler,
        private readonly sessionHandler: SessionHandler,
        private readonly logger: Logger
    ) {
        this.connections = new Map();
        this.initializeMetrics();
        this.initializeCircuitBreaker();
        this.initializeConnectionPool();
    }

    /**
     * Initializes the WebSocket server with enhanced monitoring and reliability features
     */
    public async initialize(port: number): Promise<void> {
        try {
            this.wss = new WebSocketServer({
                port,
                perMessageDeflate: true,
                maxPayload: MAX_MESSAGE_SIZE,
                clientTracking: true
            });

            this.setupServerHandlers();
            this.startHealthMonitoring();

            this.logger.info(`WebSocket server initialized on port ${port}`);
        } catch (error) {
            this.logger.error('Failed to initialize WebSocket server:', error);
            throw error;
        }
    }

    /**
     * Handles new WebSocket connections with validation and monitoring
     */
    public async handleConnection(ws: WebSocket, request: any): Promise<void> {
        const connectionId = crypto.randomUUID();
        const startTime = Date.now();

        try {
            // Apply rate limiting
            if (!this.checkRateLimit(request)) {
                throw new Error('Rate limit exceeded');
            }

            // Acquire connection from pool
            const pooledConnection = await this.connectionPool.acquire();

            // Set up connection monitoring
            this.setupConnectionMonitoring(ws, connectionId);

            // Initialize compression
            this.setupCompression(ws);

            // Track connection
            this.connections.set(connectionId, ws);
            this.connectionGauge.inc();

            // Route to appropriate handler based on path
            if (request.url.includes('/fleet')) {
                await this.fleetHandler.handleConnection(ws, connectionId);
            } else if (request.url.includes('/game')) {
                await this.gameHandler.handleGameEvent(ws, { type: 'JOIN_SESSION', payload: { sessionId: connectionId }});
            } else if (request.url.includes('/session')) {
                await this.sessionHandler.handleConnection(ws, connectionId, request.headers['x-participant-id']);
            }

            this.logger.info('New WebSocket connection established', {
                connectionId,
                setupTime: Date.now() - startTime
            });

        } catch (error) {
            this.handleError('connection', error, connectionId);
            ws.close(1011, error.message);
        }
    }

    /**
     * Routes messages with enhanced reliability and monitoring
     */
    public async handleMessage(ws: WebSocket, message: any): Promise<void> {
        const startTime = Date.now();
        const connectionId = this.getConnectionId(ws);

        try {
            // Decompress message if needed
            const decompressedMessage = await this.decompressMessage(message);

            // Validate message format
            if (!this.validateMessage(decompressedMessage)) {
                throw new Error('Invalid message format');
            }

            // Process through circuit breaker
            await this.circuitBreaker.fire(async () => {
                const parsedMessage = JSON.parse(decompressedMessage);

                // Route to appropriate handler
                if (parsedMessage.type.startsWith('FLEET_')) {
                    await this.fleetHandler.handleMessage(ws, parsedMessage);
                } else if (parsedMessage.type.startsWith('GAME_')) {
                    await this.gameHandler.handleGameEvent(ws, parsedMessage);
                } else if (parsedMessage.type.startsWith('SESSION_')) {
                    await this.sessionHandler.handleMessage(connectionId, parsedMessage);
                }
            });

            // Update metrics
            this.messageCounter.inc();
            this.latencyHistogram.observe(Date.now() - startTime);

        } catch (error) {
            this.handleError('message', error, connectionId);
        }
    }

    /**
     * Handles disconnections with enhanced cleanup and recovery
     */
    public async handleDisconnection(ws: WebSocket, code: number, reason: string): Promise<void> {
        const connectionId = this.getConnectionId(ws);

        try {
            // Release connection back to pool
            await this.connectionPool.release(ws);

            // Clean up connection tracking
            this.connections.delete(connectionId);
            this.connectionGauge.dec();

            // Notify handlers
            await this.fleetHandler.handleDisconnection(connectionId);
            await this.sessionHandler.handleDisconnection(connectionId, connectionId);

            this.logger.info('WebSocket connection closed', {
                connectionId,
                code,
                reason
            });

        } catch (error) {
            this.handleError('disconnection', error, connectionId);
        }
    }

    private initializeMetrics(): void {
        this.messageCounter = new Counter({
            name: 'websocket_messages_total',
            help: 'Total number of WebSocket messages processed'
        });

        this.connectionGauge = new Gauge({
            name: 'websocket_connections_active',
            help: 'Number of active WebSocket connections'
        });

        this.latencyHistogram = new Histogram({
            name: 'websocket_message_latency_ms',
            help: 'WebSocket message processing latency in milliseconds',
            buckets: [5, 10, 25, 50, 100, 250, 500, 1000]
        });
    }

    private initializeCircuitBreaker(): void {
        this.circuitBreaker = new CircuitBreaker({
            timeout: 5000,
            errorThreshold: CIRCUIT_BREAKER_THRESHOLD,
            resetTimeout: 30000
        });
    }

    private initializeConnectionPool(): void {
        this.connectionPool = genericPool.createPool({
            create: async () => new WebSocket(null),
            destroy: async (ws: WebSocket) => {
                ws.terminate();
            }
        }, {
            max: CONNECTION_POOL_SIZE,
            min: 10,
            testOnBorrow: true
        });
    }

    private setupServerHandlers(): void {
        this.wss.on('connection', this.handleConnection.bind(this));
        this.wss.on('error', (error) => {
            this.logger.error('WebSocket server error:', error);
        });
    }

    private setupConnectionMonitoring(ws: WebSocket, connectionId: string): void {
        let lastPing = Date.now();

        const heartbeat = setInterval(() => {
            if (Date.now() - lastPing > CONNECTION_TIMEOUT) {
                clearInterval(heartbeat);
                ws.terminate();
                return;
            }
            ws.ping();
        }, HEARTBEAT_INTERVAL);

        ws.on('pong', () => {
            lastPing = Date.now();
        });

        ws.on('close', (code, reason) => {
            clearInterval(heartbeat);
            this.handleDisconnection(ws, code, reason.toString());
        });
    }

    private startHealthMonitoring(): void {
        setInterval(() => {
            const metrics = {
                connections: this.connections.size,
                poolSize: this.connectionPool.size,
                circuitBreakerState: this.circuitBreaker.status
            };
            this.logger.debug('WebSocket health metrics:', metrics);
        }, HEALTH_CHECK_INTERVAL);
    }

    private async decompressMessage(message: any): Promise<string> {
        return new Promise((resolve, reject) => {
            if (message instanceof Buffer) {
                zlib.inflate(message, (error, result) => {
                    if (error) reject(error);
                    else resolve(result.toString());
                });
            } else {
                resolve(message.toString());
            }
        });
    }

    private validateMessage(message: string): boolean {
        try {
            const parsed = JSON.parse(message);
            return parsed && typeof parsed === 'object' && 'type' in parsed;
        } catch {
            return false;
        }
    }

    private checkRateLimit(request: any): boolean {
        // Rate limiting implementation would go here
        return true;
    }

    private setupCompression(ws: WebSocket): void {
        ws.on('message', (data) => {
            zlib.inflate(data, (error, result) => {
                if (!error) {
                    this.handleMessage(ws, result);
                }
            });
        });
    }

    private getConnectionId(ws: WebSocket): string {
        for (const [id, conn] of this.connections) {
            if (conn === ws) return id;
        }
        return '';
    }

    private handleError(context: string, error: Error, connectionId?: string): void {
        this.logger.error(`WebSocket error in ${context}:`, {
            error: error.message,
            connectionId,
            stack: error.stack
        });
    }
}