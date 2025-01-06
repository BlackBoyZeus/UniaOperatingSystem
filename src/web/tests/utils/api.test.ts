// External imports with versions for security tracking
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals'; // ^29.5.0
import axios from 'axios'; // ^1.4.0
import { rest } from 'msw'; // ^1.2.0

// Internal imports
import { createRequestConfig, handleApiError, createWebSocketUrl } from '../../src/utils/api.utils';
import { server } from '../mocks/server';

// Constants for testing
const MOCK_BASE_URL = 'http://localhost:3000';
const MOCK_AUTH_TOKEN = 'test-auth-token';
const TEST_TIMEOUT = 5000;
const FLEET_SIZE_LIMIT = 32;
const MAX_LATENCY = 50;
const SECURITY_PROTOCOL_VERSION = 'TLS_1_3';
const RETRY_ATTEMPTS = 3;
const ERROR_SAMPLE_RATE = 0.1;

describe('createRequestConfig', () => {
    beforeEach(() => {
        server.listen();
    });

    afterEach(() => {
        server.resetHandlers();
        server.close();
    });

    it('should create request config with security headers and hardware token', async () => {
        const hardwareToken = 'mock-hardware-token';
        const options = {
            endpoint: '/test',
            method: 'POST',
            timeout: TEST_TIMEOUT
        };

        const config = await createRequestConfig(options, hardwareToken);

        expect(config).toMatchObject({
            timeout: TEST_TIMEOUT,
            headers: {
                'X-Hardware-Token': hardwareToken,
                'X-Security-Version': SECURITY_PROTOCOL_VERSION
            }
        });
        expect(config.httpsAgent).toBeDefined();
    });

    it('should configure retry policy with exponential backoff', async () => {
        const config = await createRequestConfig({
            endpoint: '/test',
            method: 'GET'
        }, 'mock-hardware-token');

        expect(config.retryConfig).toMatchObject({
            retries: RETRY_ATTEMPTS,
            retryCondition: expect.any(Function),
            retryDelay: expect.any(Function)
        });
    });

    it('should enforce TLS 1.3 with certificate pinning', async () => {
        const config = await createRequestConfig({
            endpoint: '/test',
            method: 'GET'
        }, 'mock-hardware-token');

        expect(config.httpsAgent.options).toMatchObject({
            minVersion: SECURITY_PROTOCOL_VERSION,
            rejectUnauthorized: true
        });
    });

    it('should validate hardware token before creating config', async () => {
        await expect(createRequestConfig({
            endpoint: '/test'
        }, 'invalid-token')).rejects.toThrow('Invalid hardware token');
    });

    it('should configure compression options for performance', async () => {
        const config = await createRequestConfig({
            endpoint: '/test'
        }, 'mock-hardware-token');

        expect(config.decompress).toBe(true);
        expect(config.headers['Accept-Encoding']).toBe('gzip, deflate, br');
    });
});

describe('handleApiError', () => {
    it('should handle network errors with security context', () => {
        const networkError = new axios.AxiosError(
            'Network Error',
            'ECONNABORTED',
            {
                headers: {
                    'X-Security-Version': SECURITY_PROTOCOL_VERSION
                }
            }
        );

        const result = handleApiError(networkError, {
            hardwareToken: 'mock-token',
            securityLevel: 'HIGH',
            timestamp: Date.now()
        });

        expect(result).toMatchObject({
            message: 'Network Error',
            code: 500,
            context: {
                security: expect.any(Object)
            }
        });
    });

    it('should detect and log security breaches', () => {
        const securityError = new axios.AxiosError(
            'Invalid Certificate',
            'CERT_INVALID',
            {
                headers: {
                    'X-Security-Version': 'TLS_1_2' // Downgrade attempt
                }
            }
        );

        const consoleSpy = jest.spyOn(console, 'error');
        handleApiError(securityError);

        expect(consoleSpy).toHaveBeenCalledWith(
            'Security breach detected:',
            expect.any(Object)
        );
    });

    it('should preserve fleet coordination context in errors', () => {
        const fleetError = new axios.AxiosError(
            'Fleet Sync Failed',
            'SYNC_ERROR',
            {
                headers: {
                    'X-Fleet-ID': 'test-fleet'
                }
            }
        );

        const result = handleApiError(fleetError);

        expect(result.context).toMatchObject({
            method: expect.any(String),
            timestamp: expect.any(String)
        });
    });

    it('should sample errors based on configured rate', () => {
        const consoleSpy = jest.spyOn(console, 'debug');
        const error = new axios.AxiosError('Test Error');

        // Force sampling by mocking Math.random
        const originalRandom = Math.random;
        Math.random = jest.fn().mockReturnValue(ERROR_SAMPLE_RATE - 0.01);

        handleApiError(error);

        expect(consoleSpy).toHaveBeenCalledWith(
            'API Error Sample:',
            expect.any(Object)
        );

        Math.random = originalRandom;
    });
});

describe('createWebSocketUrl', () => {
    it('should generate secure WebSocket URL with fleet parameters', () => {
        const endpoint = '/fleet/sync';
        const options = {
            hardwareToken: 'mock-token',
            compression: true,
            poolSize: FLEET_SIZE_LIMIT
        };

        const url = createWebSocketUrl(endpoint, options);

        expect(url).toMatch(/^wss?:/);
        expect(url).toContain('version=' + SECURITY_PROTOCOL_VERSION);
        expect(url).toContain('compression=1');
        expect(url).toContain('pool_size=' + FLEET_SIZE_LIMIT);
    });

    it('should configure connection pooling for fleet coordination', () => {
        const url = createWebSocketUrl('/fleet/sync', {
            hardwareToken: 'mock-token',
            poolSize: 4,
            poolStrategy: 'round-robin'
        });

        expect(url).toContain('pool_size=4');
        expect(url).toContain('pool_strategy=round-robin');
    });

    it('should include hardware token and security signature', () => {
        const url = createWebSocketUrl('/test', {
            hardwareToken: 'mock-token'
        });

        expect(url).toContain('hw=mock-token');
        expect(url).toMatch(/sig=[a-zA-Z0-9-_]+/);
    });

    it('should enforce security protocol version', () => {
        const url = createWebSocketUrl('/test', {
            hardwareToken: 'mock-token'
        });

        expect(url).toContain(`version=${SECURITY_PROTOCOL_VERSION}`);
    });
});