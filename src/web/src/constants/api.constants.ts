/**
 * API Constants for TALD UNIA Web Frontend
 * Defines core API endpoints, timeouts, retry policies and request configurations
 * for communication with backend services.
 * 
 * @version 1.0.0
 */

/**
 * API endpoint constants for all backend service routes
 */
export const API_ENDPOINTS = {
  FLEET: {
    JOIN: '/fleet/join',
    SYNC: '/fleet/sync',
    LIST: '/fleet/list',
    CREATE: '/fleet/create', 
    LEAVE: '/fleet/leave',
    STATUS: '/fleet/status',
    MEMBERS: '/fleet/members',
    INVITE: '/fleet/invite'
  },
  GAME: {
    STATE: '/game/state',
    SESSION: '/game/session',
    SYNC: '/game/sync',
    SAVE: '/game/save',
    LOAD: '/game/load',
    METRICS: '/game/metrics'
  },
  LIDAR: {
    SCAN: '/lidar/scan',
    UPLOAD: '/lidar/upload',
    PROCESS: '/lidar/process',
    CALIBRATE: '/lidar/calibrate',
    QUALITY: '/lidar/quality'
  },
  USER: {
    AUTH: '/user/auth',
    PROFILE: '/user/profile',
    SETTINGS: '/user/settings',
    PREFERENCES: '/user/preferences',
    DEVICES: '/user/devices'
  }
} as const;

/**
 * API timeout constants in milliseconds
 * Configured for sub-50ms network latency requirements
 */
export const API_TIMEOUTS = {
  DEFAULT: 5000,    // Default timeout for API requests
  UPLOAD: 30000,    // Extended timeout for LiDAR data uploads
  SYNC: 1000,       // Aggressive timeout for state sync
  SCAN: 3000,       // Timeout for LiDAR scanning
  PROCESS: 10000    // Timeout for heavy processing operations
} as const;

/**
 * Retry policy configuration with exponential backoff
 * Ensures resilient network communication
 */
export const RETRY_POLICY = {
  MAX_RETRIES: 3,           // Maximum number of retry attempts
  BACKOFF_FACTOR: 1.5,      // Exponential backoff multiplier
  INITIAL_DELAY: 1000,      // Initial retry delay in ms
  MAX_DELAY: 8000,          // Maximum retry delay in ms
  JITTER: 0.1              // Random jitter factor for retry timing
} as const;

/**
 * WebSocket configuration for real-time game state synchronization
 * Optimized for 32-device fleet management
 */
export const WEBSOCKET_CONFIG = {
  RECONNECT_INTERVAL: 1000,      // Reconnection attempt interval in ms
  MAX_RECONNECT_ATTEMPTS: 5,     // Maximum reconnection attempts
  HEARTBEAT_INTERVAL: 30000,     // Heartbeat interval in ms
  PING_TIMEOUT: 5000,           // Ping timeout threshold in ms
  CLOSE_TIMEOUT: 3000           // Connection close timeout in ms
} as const;

/**
 * Standard HTTP headers for API communication
 */
export const HTTP_HEADERS = {
  ACCEPT: 'application/json',
  CONTENT_TYPE: 'application/json',
  CACHE_CONTROL: 'no-cache',
  PRAGMA: 'no-cache'
} as const;