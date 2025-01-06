import { injectable, inject } from 'tsyringe';
import WebSocket from 'ws'; // v8.13.0
import { Logger } from 'winston'; // v3.10.0
import CircuitBreaker from 'opossum'; // v6.0.0
import { MetricsService } from '@metrics/service'; // v1.0.0

import { FleetService } from '../../services/fleet/FleetService';
import { 
    IFleetMember, 
    FleetStatus, 
    IFleetNetworkStats,
    IFleetSecurity 
} from '../../interfaces/fleet.interface';

// Constants for fleet management and performance monitoring
const FLEET_SYNC_INTERVAL = 50; // 50ms sync interval
const MAX_FLEET_SIZE = 32;
const HEARTBEAT_INTERVAL = 30000; // 30 seconds
const MAX_RETRY_ATTEMPTS = 3;
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_MESSAGES_PER_WINDOW = 1000;
const CIRCUIT_BREAKER_TIMEOUT = 5000; // 5 seconds

interface ConnectionMetrics {
    messageCount: number;
    lastMessageTime: number;
    latency: number;
    errors: number;
    windowStart: number;
}

@injectable()
export class FleetWebSocketHandler {
    private readonly fleetConnections: Map<string, WebSocket>;
    private readonly connectionMetrics: Map<string, ConnectionMetrics>;
    private readonly connectionBreaker: CircuitBreaker;
    private readonly rateLimiters: Map<string, number>;

    constructor(
        @inject('FleetService') private readonly fleetService: FleetService,
        @inject('Logger') private readonly logger: Logger,
        @inject('MetricsService') private readonly metricsService: MetricsService
    ) {
        this.fleetConnections = new Map();
        this.connectionMetrics = new Map();
        this.rateLimiters = new Map();

        // Initialize circuit breaker for connection handling
        this.connectionBreaker = new CircuitBreaker(this.executeWithRetry.bind(this), {
            timeout: CIRCUIT_BREAKER_TIMEOUT,
            resetTimeout: CIRCUIT_BREAKER_TIMEOUT * 2,
            errorThresholdPercentage: 50
        });

        this.startHealthCheck();
    }

    /**
     * Handles new WebSocket connections with validation and monitoring
     */
    public async handleConnection(ws: WebSocket, deviceId: string): Promise<void> {
        try {
            // Validate fleet capacity
            if (this.fleetConnections.size >= MAX_FLEET_SIZE) {
                throw new Error(`Fleet size limit (${MAX_FLEET_SIZE}) reached`);
            }

            // Initialize connection metrics
            this.connectionMetrics.set(deviceId, {
                messageCount: 0,
                lastMessageTime: Date.now(),
                latency: 0,
                errors: 0,
                windowStart: Date.now()
            });

            // Set up WebSocket event handlers
            this.setupWebSocketHandlers(ws, deviceId);

            // Add to connection pool
            this.fleetConnections.set(deviceId, ws);

            // Start heartbeat monitoring
            this.startHeartbeat(ws, deviceId);

            this.logger.info('Device connected to fleet', { deviceId });
            this.metricsService.incrementCounter('fleet.connections.active');

        } catch (error) {
            this.handleError('handleConnection', error, deviceId);
            ws.close(1011, error.message);
        }
    }

    /**
     * Handles incoming WebSocket messages with validation and rate limiting
     */
    public async handleMessage(ws: WebSocket, message: any): Promise<void> {
        const startTime = Date.now();
        const deviceId = this.getDeviceId(ws);

        try {
            // Validate rate limits
            if (!this.checkRateLimit(deviceId)) {
                throw new Error('Rate limit exceeded');
            }

            // Validate message format
            if (!this.validateMessage(message)) {
                throw new Error('Invalid message format');
            }

            // Process message based on type
            await this.connectionBreaker.fire(async () => {
                switch (message.type) {
                    case 'JOIN_FLEET':
                        await this.handleJoinFleet(deviceId, message.payload);
                        break;
                    case 'SYNC_STATE':
                        await this.handleStateSync(deviceId, message.payload);
                        break;
                    case 'LEAVE_FLEET':
                        await this.handleLeaveFleet(deviceId);
                        break;
                    default:
                        throw new Error('Unknown message type');
                }
            });

            // Update metrics
            this.updateMetrics(deviceId, startTime);

        } catch (error) {
            this.handleError('handleMessage', error, deviceId);
            this.sendErrorResponse(ws, error.message);
        }
    }

    /**
     * Sets up WebSocket event handlers for a connection
     */
    private setupWebSocketHandlers(ws: WebSocket, deviceId: string): void {
        ws.on('message', async (data: WebSocket.Data) => {
            try {
                const message = JSON.parse(data.toString());
                await this.handleMessage(ws, message);
            } catch (error) {
                this.handleError('messageHandler', error, deviceId);
            }
        });

        ws.on('close', () => {
            this.handleDisconnection(deviceId);
        });

        ws.on('error', (error) => {
            this.handleError('websocketError', error, deviceId);
        });
    }

    /**
     * Handles fleet join requests
     */
    private async handleJoinFleet(deviceId: string, payload: any): Promise<void> {
        await this.fleetService.joinFleet(payload.fleetId, {
            id: deviceId,
            status: FleetStatus.CONNECTING,
            capabilities: payload.capabilities,
            joinedAt: Date.now()
        });
    }

    /**
     * Handles state synchronization requests
     */
    private async handleStateSync(deviceId: string, payload: any): Promise<void> {
        await this.fleetService.synchronizeFleetState(payload.fleetId, payload.state);
    }

    /**
     * Handles fleet leave requests
     */
    private async handleLeaveFleet(deviceId: string): Promise<void> {
        const ws = this.fleetConnections.get(deviceId);
        if (ws) {
            await this.fleetService.leaveFleet(deviceId);
            ws.close(1000, 'Left fleet');
        }
    }

    /**
     * Implements rate limiting for message handling
     */
    private checkRateLimit(deviceId: string): boolean {
        const metrics = this.connectionMetrics.get(deviceId);
        if (!metrics) return false;

        const now = Date.now();
        if (now - metrics.windowStart > RATE_LIMIT_WINDOW) {
            metrics.messageCount = 0;
            metrics.windowStart = now;
        }

        metrics.messageCount++;
        return metrics.messageCount <= MAX_MESSAGES_PER_WINDOW;
    }

    /**
     * Updates connection metrics
     */
    private updateMetrics(deviceId: string, startTime: number): void {
        const metrics = this.connectionMetrics.get(deviceId);
        if (metrics) {
            metrics.latency = Date.now() - startTime;
            metrics.lastMessageTime = Date.now();

            this.metricsService.recordMetric('fleet.message.latency', metrics.latency);
        }
    }

    /**
     * Starts periodic health checks for all connections
     */
    private startHealthCheck(): void {
        setInterval(() => {
            this.fleetConnections.forEach((ws, deviceId) => {
                const metrics = this.connectionMetrics.get(deviceId);
                if (metrics && Date.now() - metrics.lastMessageTime > HEARTBEAT_INTERVAL) {
                    this.handleDisconnection(deviceId);
                }
            });
        }, HEARTBEAT_INTERVAL);
    }

    /**
     * Handles connection heartbeat
     */
    private startHeartbeat(ws: WebSocket, deviceId: string): void {
        const interval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.ping();
            } else {
                clearInterval(interval);
            }
        }, HEARTBEAT_INTERVAL / 2);
    }

    /**
     * Handles device disconnections
     */
    private async handleDisconnection(deviceId: string): Promise<void> {
        try {
            await this.fleetService.leaveFleet(deviceId);
            this.fleetConnections.delete(deviceId);
            this.connectionMetrics.delete(deviceId);
            this.metricsService.decrementCounter('fleet.connections.active');
            this.logger.info('Device disconnected from fleet', { deviceId });
        } catch (error) {
            this.handleError('handleDisconnection', error, deviceId);
        }
    }

    /**
     * Executes operations with retry mechanism
     */
    private async executeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
        let lastError: Error | null = null;
        for (let attempt = 1; attempt <= MAX_RETRY_ATTEMPTS; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error;
                await new Promise(resolve => 
                    setTimeout(resolve, Math.pow(2, attempt) * 100)
                );
            }
        }
        throw lastError;
    }

    /**
     * Validates message format and content
     */
    private validateMessage(message: any): boolean {
        return message && 
               typeof message === 'object' && 
               typeof message.type === 'string' && 
               message.payload !== undefined;
    }

    /**
     * Gets device ID from WebSocket connection
     */
    private getDeviceId(ws: WebSocket): string {
        for (const [deviceId, socket] of this.fleetConnections.entries()) {
            if (socket === ws) return deviceId;
        }
        throw new Error('Device ID not found');
    }

    /**
     * Sends error response to client
     */
    private sendErrorResponse(ws: WebSocket, message: string): void {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'ERROR',
                payload: { message }
            }));
        }
    }

    /**
     * Handles and logs errors with metrics tracking
     */
    private handleError(context: string, error: Error, deviceId?: string): void {
        this.logger.error(`Fleet WebSocket error in ${context}`, {
            error: error.message,
            deviceId,
            stack: error.stack
        });

        if (deviceId) {
            const metrics = this.connectionMetrics.get(deviceId);
            if (metrics) {
                metrics.errors++;
                this.metricsService.incrementCounter('fleet.errors');
            }
        }
    }
}