import axios, { AxiosRequestConfig } from 'axios';

// Base configuration
export const BASE_URL = process.env.VITE_API_BASE_URL || 'https://api.tald.unia';
export const API_VERSION = 'v1';
export const DEFAULT_TIMEOUT = 5000;
export const MAX_FLEET_SIZE = 32;
export const MAX_LATENCY = 50;

// API Endpoints configuration
export const ENDPOINTS = {
  FLEET: {
    JOIN: '/fleet/join',
    SYNC: '/fleet/sync',
    LIST: '/fleet/list',
    CREATE: '/fleet/create',
    LEAVE: '/fleet/leave',
    STATUS: '/fleet/status',
    MEMBERS: '/fleet/members',
    HEARTBEAT: '/fleet/heartbeat'
  },
  GAME: {
    STATE: '/game/state',
    SESSION: '/game/session',
    SYNC: '/game/sync',
    POSITION: '/game/position',
    INTERACTION: '/game/interaction',
    EVENTS: '/game/events'
  },
  LIDAR: {
    SCAN: '/lidar/scan',
    UPLOAD: '/lidar/upload',
    PROCESS: '/lidar/process',
    MESH: '/lidar/mesh',
    CALIBRATE: '/lidar/calibrate',
    OPTIMIZE: '/lidar/optimize'
  },
  REALTIME: {
    CONNECT: '/rt/connect',
    SUBSCRIBE: '/rt/subscribe',
    PUBLISH: '/rt/publish',
    PRESENCE: '/rt/presence'
  }
} as const;

// Timeout configurations
export const TIMEOUTS = {
  DEFAULT: 5000,
  UPLOAD: 30000,
  SYNC: 1000,
  FLEET: 2000,
  LIDAR: 15000,
  WEBSOCKET: 45000
} as const;

// Retry policy configuration
export const RETRY_POLICY = {
  MAX_RETRIES: 3,
  BACKOFF_FACTOR: 1.5,
  INITIAL_DELAY: 1000,
  MAX_DELAY: 5000,
  JITTER: 100,
  TIMEOUT_MULTIPLIER: 1.5,
  CIRCUIT_BREAKER: {
    FAILURE_THRESHOLD: 5,
    RESET_TIMEOUT: 30000
  }
} as const;

// WebSocket configuration
export const WEBSOCKET = {
  RECONNECT_INTERVAL: 1000,
  MAX_RECONNECT_ATTEMPTS: 5,
  HEARTBEAT_INTERVAL: 30000,
  PING_TIMEOUT: 5000,
  POOL_SIZE: 4,
  BATCH_INTERVAL: 50,
  COMPRESSION: true,
  PROTOCOL_VERSION: '1.0'
} as const;

// Security configuration
export const SECURITY = {
  RATE_LIMIT: {
    MAX_REQUESTS: 100,
    WINDOW_MS: 60000
  },
  CORS: {
    ALLOWED_ORIGINS: ['https://*.tald.unia'],
    ALLOWED_METHODS: ['GET', 'POST', 'PUT', 'DELETE'],
    ALLOW_CREDENTIALS: true
  },
  REQUEST_SIGNING: {
    ALGORITHM: 'SHA-256',
    EXPIRY: 300000
  }
} as const;

// Monitoring configuration
export const MONITORING = {
  METRICS_INTERVAL: 10000,
  HEALTH_CHECK_INTERVAL: 30000,
  TELEMETRY_BATCH_SIZE: 100,
  ERROR_SAMPLING_RATE: 0.1
} as const;

// Main API configuration object
export const apiConfig = {
  baseUrl: BASE_URL,
  apiVersion: API_VERSION,
  timeout: DEFAULT_TIMEOUT,
  retryPolicy: RETRY_POLICY,
  endpoints: ENDPOINTS,
  websocket: WEBSOCKET,
  security: SECURITY,
  monitoring: MONITORING,

  // Request configuration factory
  getRequestConfig(endpoint: string, options: Partial<AxiosRequestConfig> = {}): AxiosRequestConfig {
    return {
      baseURL: `${this.baseUrl}/${this.apiVersion}`,
      timeout: options.timeout || this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Version': this.apiVersion,
        ...options.headers
      },
      ...options
    };
  },

  // WebSocket URL generator
  getWebSocketUrl(endpoint: string): string {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${wsProtocol}//${this.baseUrl.replace(/^https?:\/\//, '')}/${this.apiVersion}${endpoint}`;
  },

  // Fleet-specific configuration
  fleet: {
    maxSize: MAX_FLEET_SIZE,
    maxLatency: MAX_LATENCY,
    heartbeatInterval: WEBSOCKET.HEARTBEAT_INTERVAL,
    syncInterval: TIMEOUTS.SYNC
  },

  // LiDAR-specific configuration
  lidar: {
    uploadTimeout: TIMEOUTS.UPLOAD,
    processTimeout: TIMEOUTS.LIDAR,
    compressionEnabled: WEBSOCKET.COMPRESSION
  }
} as const;

// Type definitions for configuration consumers
export type ApiConfig = typeof apiConfig;
export type Endpoints = typeof ENDPOINTS;
export type RetryPolicy = typeof RETRY_POLICY;
export type WebSocketConfig = typeof WEBSOCKET;
export type SecurityConfig = typeof SECURITY;
export type MonitoringConfig = typeof MONITORING;

export default apiConfig;