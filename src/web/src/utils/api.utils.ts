// External imports with versions for security tracking
import axios, { AxiosRequestConfig, AxiosError } from 'axios'; // ^1.4.0
import axiosRetry from 'axios-retry'; // ^3.5.0
import CircuitBreaker from 'opossum'; // ^7.1.0
import { SecurityUtils } from '@tald/security-utils'; // ^2.0.0

// Internal imports
import { apiConfig } from '../config/api.config';
import { parseToken } from './auth.utils';

// Constants for API utilities
const DEFAULT_TIMEOUT = 5000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;
const CIRCUIT_BREAKER_THRESHOLD = 5;
const ERROR_SAMPLING_RATE = 0.1;
const SECURITY_PROTOCOL_VERSION = 'TLSv1.3';
const CERTIFICATE_ROTATION_INTERVAL = 86400000; // 24 hours

// Circuit breaker configuration
const breaker = new CircuitBreaker(axios, {
    timeout: DEFAULT_TIMEOUT,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
});

// Enhanced security utilities initialization
const securityUtils = new SecurityUtils({
    protocolVersion: SECURITY_PROTOCOL_VERSION,
    certificateRotationInterval: CERTIFICATE_ROTATION_INTERVAL,
    requireHardwareBackedKeys: true
});

/**
 * Creates an enhanced request configuration with advanced security features
 * @param options - Request configuration options
 * @param hardwareToken - Hardware-specific security token
 * @returns Enhanced axios request configuration
 */
export const createRequestConfig = async (
    options: RequestOptions,
    hardwareToken: string
): Promise<AxiosRequestConfig> => {
    try {
        // Validate hardware token
        const isValidToken = await securityUtils.validateHardwareToken(hardwareToken);
        if (!isValidToken) {
            throw new Error('Invalid hardware token');
        }

        // Merge with default configuration
        const config: AxiosRequestConfig = {
            ...apiConfig.getRequestConfig(options.endpoint),
            ...options,
            timeout: options.timeout || DEFAULT_TIMEOUT,
            headers: {
                ...options.headers,
                'X-Hardware-Token': hardwareToken,
                'X-Security-Version': SECURITY_PROTOCOL_VERSION,
                'X-Request-Signature': await securityUtils.signRequest(options)
            }
        };

        // Configure retry mechanism
        axiosRetry(axios, {
            retries: MAX_RETRIES,
            retryDelay: (retryCount) => {
                return retryCount * RETRY_DELAY;
            },
            retryCondition: (error) => {
                return axiosRetry.isNetworkOrIdempotentRequestError(error) &&
                    !error.response?.status?.toString().startsWith('4');
            }
        });

        // Apply certificate pinning
        config.httpsAgent = securityUtils.createHttpsAgent({
            ...apiConfig.security,
            rejectUnauthorized: true,
            minVersion: SECURITY_PROTOCOL_VERSION
        });

        // Configure compression
        config.decompress = true;
        config.headers['Accept-Encoding'] = 'gzip, deflate, br';

        return config;
    } catch (error) {
        console.error('Failed to create request config:', error);
        throw error;
    }
};

/**
 * Enhanced error handling with security breach detection
 * @param error - Axios error object
 * @param securityContext - Security context for error analysis
 * @returns Enhanced error object with security context
 */
export const handleApiError = (
    error: AxiosError,
    securityContext?: SecurityContext
): ApiError => {
    // Sample errors for monitoring
    if (Math.random() < ERROR_SAMPLING_RATE) {
        console.debug('API Error Sample:', {
            url: error.config?.url,
            method: error.config?.method,
            status: error.response?.status,
            timestamp: new Date().toISOString()
        });
    }

    // Analyze for security breaches
    const securityAnalysis = securityUtils.analyzeError(error, securityContext);
    if (securityAnalysis.breachDetected) {
        console.error('Security breach detected:', securityAnalysis);
        // Trigger security alert
        securityUtils.triggerSecurityAlert(securityAnalysis);
    }

    return {
        message: error.response?.data?.message || error.message,
        code: error.response?.status || 500,
        context: {
            url: error.config?.url,
            method: error.config?.method,
            timestamp: new Date().toISOString(),
            security: securityAnalysis
        }
    };
};

/**
 * Generates enhanced WebSocket URL with security features
 * @param endpoint - WebSocket endpoint
 * @param options - Connection options
 * @returns Secure WebSocket URL
 */
export const createWebSocketUrl = (
    endpoint: string,
    options: ConnectionOptions
): string => {
    const baseUrl = apiConfig.getWebSocketUrl(endpoint);
    const securityParams = new URLSearchParams({
        version: SECURITY_PROTOCOL_VERSION,
        compression: options.compression ? '1' : '0',
        pool: options.poolSize?.toString() || '1'
    });

    // Add enhanced security parameters
    securityParams.append('sig', securityUtils.generateConnectionSignature(endpoint));
    securityParams.append('hw', options.hardwareToken);

    // Configure connection pooling
    if (options.poolSize && options.poolSize > 1) {
        securityParams.append('pool_size', options.poolSize.toString());
        securityParams.append('pool_strategy', options.poolStrategy || 'round-robin');
    }

    return `${baseUrl}?${securityParams.toString()}`;
};

// Type definitions
interface RequestOptions extends Partial<AxiosRequestConfig> {
    endpoint: string;
    hardwareToken?: string;
}

interface SecurityContext {
    hardwareToken: string;
    securityLevel: string;
    timestamp: number;
}

interface ApiError {
    message: string;
    code: number;
    context: {
        url?: string;
        method?: string;
        timestamp: string;
        security?: any;
    };
}

interface ConnectionOptions {
    hardwareToken: string;
    compression?: boolean;
    poolSize?: number;
    poolStrategy?: 'round-robin' | 'least-connections';
}