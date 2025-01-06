// External imports - versions specified for security tracking
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'; // ^1.4.0
import { Socket, io } from 'socket.io-client'; // ^4.7.0
import { CircuitBreaker } from 'opossum'; // ^7.1.0
import { compress, decompress } from 'lz4-js'; // ^0.4.1
import { v4 as uuidv4 } from 'uuid'; // ^9.0.0

// Internal imports
import { apiConfig } from '../config/api.config';
import { getStoredAuthToken } from '../utils/auth.utils';
import { DeviceCapabilityType, UserStatusType } from '../interfaces/user.interface';

// Monitoring decorator
function monitor(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = async function(...args: any[]) {
        const startTime = performance.now();
        try {
            const result = await originalMethod.apply(this, args);
            this.recordMetric(propertyKey, performance.now() - startTime);
            return result;
        } catch (error) {
            this.recordError(propertyKey, error);
            throw error;
        }
    };
    return descriptor;
}

// Rate limiting decorator
function rateLimit(limit: number = 100, window: number = 60000) {
    return function(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
        const requests = new Map<string, number[]>();
        const originalMethod = descriptor.value;
        
        descriptor.value = async function(...args: any[]) {
            const now = Date.now();
            const key = `${propertyKey}_${now}`;
            
            const windowRequests = requests.get(key) || [];
            const validRequests = windowRequests.filter(time => now - time < window);
            
            if (validRequests.length >= limit) {
                throw new Error('Rate limit exceeded');
            }
            
            validRequests.push(now);
            requests.set(key, validRequests);
            
            return originalMethod.apply(this, args);
        };
        return descriptor;
    };
}

@injectable()
export class ApiService {
    private api: AxiosInstance;
    private socket: Socket | null = null;
    private breaker: CircuitBreaker;
    private connectionPool: Map<string, Socket> = new Map();
    private metrics: Map<string, number[]> = new Map();
    private deviceCapabilities: DeviceCapabilityType | null = null;

    constructor() {
        this.api = this.createApiInstance();
        this.breaker = this.createCircuitBreaker();
        this.initializeWebSocket();
        this.setupMetricsCollection();
    }

    private createApiInstance(): AxiosInstance {
        const instance = axios.create({
            baseURL: `${apiConfig.baseUrl}/${apiConfig.apiVersion}`,
            timeout: apiConfig.timeout,
            headers: {
                'Content-Type': 'application/json',
                'X-API-Version': apiConfig.apiVersion
            }
        });

        instance.interceptors.request.use(async (config) => {
            const auth = await getStoredAuthToken();
            if (auth) {
                config.headers.Authorization = `Bearer ${auth.accessToken}`;
                config.headers['X-Hardware-Token'] = auth.hardwareToken;
            }
            return config;
        });

        instance.interceptors.response.use(
            (response) => {
                this.recordMetric('apiResponse', response.config.duration);
                return response;
            },
            async (error) => {
                this.recordError('apiError', error);
                if (error.response?.status === 401) {
                    // Handle token refresh
                }
                throw error;
            }
        );

        return instance;
    }

    private createCircuitBreaker(): CircuitBreaker {
        return new CircuitBreaker(async (request: () => Promise<any>) => {
            return await request();
        }, {
            timeout: apiConfig.retryPolicy.MAX_DELAY,
            resetTimeout: apiConfig.retryPolicy.CIRCUIT_BREAKER.RESET_TIMEOUT,
            errorThresholdPercentage: 50,
            volumeThreshold: 10
        });
    }

    private async initializeWebSocket() {
        const auth = await getStoredAuthToken();
        if (!auth) return;

        this.socket = io(apiConfig.baseUrl, {
            transports: ['websocket'],
            auth: {
                token: auth.accessToken,
                hardwareToken: auth.hardwareToken
            },
            reconnection: true,
            reconnectionDelay: apiConfig.websocket.RECONNECT_INTERVAL,
            reconnectionAttempts: apiConfig.websocket.MAX_RECONNECT_ATTEMPTS
        });

        this.setupWebSocketHandlers();
    }

    private setupWebSocketHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            this.recordMetric('wsConnect', 1);
            this.negotiateCapabilities();
        });

        this.socket.on('disconnect', (reason) => {
            this.recordMetric('wsDisconnect', 1);
            if (reason === 'io server disconnect') {
                this.socket?.connect();
            }
        });

        this.socket.on('fleet:state', (data: any) => {
            const decompressed = decompress(data);
            this.handleFleetState(JSON.parse(decompressed));
        });

        // Setup heartbeat
        setInterval(() => {
            this.socket?.emit('heartbeat', { timestamp: Date.now() });
        }, apiConfig.websocket.HEARTBEAT_INTERVAL);
    }

    @monitor
    @rateLimit()
    public async request<T>(config: AxiosRequestConfig): Promise<T> {
        return this.breaker.fire(async () => {
            const response = await this.api.request<T>({
                ...config,
                headers: {
                    ...config.headers,
                    'X-Request-ID': uuidv4()
                }
            });
            return response.data;
        });
    }

    @monitor
    public async emit(event: string, data: any): Promise<void> {
        if (!this.socket?.connected) {
            throw new Error('WebSocket not connected');
        }

        const compressed = compress(JSON.stringify(data));
        return new Promise((resolve, reject) => {
            this.socket?.emit(event, compressed, (error: any) => {
                if (error) {
                    this.recordError('wsEmit', error);
                    reject(error);
                } else {
                    this.recordMetric('wsEmit', 1);
                    resolve();
                }
            });
        });
    }

    @monitor
    private async negotiateCapabilities(): Promise<void> {
        try {
            const response = await this.request<DeviceCapabilityType>({
                url: apiConfig.endpoints.FLEET.STATUS,
                method: 'GET'
            });
            this.deviceCapabilities = response;
            this.emit('capabilities:update', response);
        } catch (error) {
            this.recordError('capabilitiesNegotiation', error);
        }
    }

    private recordMetric(name: string, value: number) {
        const metrics = this.metrics.get(name) || [];
        metrics.push(value);
        this.metrics.set(name, metrics.slice(-100)); // Keep last 100 measurements
    }

    private recordError(context: string, error: any) {
        console.error(`[${context}] ${error.message}`, {
            timestamp: new Date().toISOString(),
            context,
            error
        });
    }

    public getMetrics() {
        const result: Record<string, { avg: number; p95: number }> = {};
        
        this.metrics.forEach((values, key) => {
            const sorted = [...values].sort((a, b) => a - b);
            result[key] = {
                avg: values.reduce((a, b) => a + b, 0) / values.length,
                p95: sorted[Math.floor(sorted.length * 0.95)]
            };
        });
        
        return result;
    }

    public async dispose() {
        this.socket?.disconnect();
        this.connectionPool.forEach(socket => socket.disconnect());
        this.connectionPool.clear();
        this.metrics.clear();
    }
}

export default ApiService;