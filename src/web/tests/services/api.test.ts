import { describe, test, expect, beforeAll, afterAll, jest } from '@jest/globals';
import { rest } from 'msw';
import { ApiService } from '../../src/services/api.service';
import { server } from '../mocks/server';
import { FleetStatus, FleetRole } from '../../src/types/fleet.types';
import { UserRoleType } from '../../src/interfaces/user.interface';
import { apiConfig } from '../../src/config/api.config';

// Mock hardware token for testing
const mockHardwareToken = {
    deviceId: '123e4567-e89b-12d3-a456-426614174000',
    signature: 'mock-signature',
    timestamp: Date.now(),
    capabilities: {
        lidarSupported: true,
        meshNetworkSupported: true,
        vulkanVersion: '1.3',
        hardwareSecurityLevel: 'HIGH',
        scanningResolution: 0.01,
        maxFleetSize: 32
    }
};

// Mock fleet response data
const mockFleetResponse = {
    fleetId: '123e4567-e89b-12d3-a456-426614174001',
    members: [],
    status: FleetStatus.ACTIVE,
    leaderToken: 'mock-leader-token',
    crdt: {}
};

// Mock game state data
const mockGameState = {
    gameId: '123e4567-e89b-12d3-a456-426614174002',
    state: { position: { x: 0, y: 0, z: 0 } },
    timestamp: Date.now(),
    version: 1,
    conflicts: []
};

// Mock LiDAR data
const mockLidarData = {
    scanId: '123e4567-e89b-12d3-a456-426614174003',
    pointCloud: new Float32Array(1000),
    metadata: { resolution: 0.01, range: 5 },
    compression: 'lz4',
    chunks: 1
};

describe('ApiService Integration Tests', () => {
    let apiService: ApiService;

    beforeAll(async () => {
        // Start mock server
        server.listen();

        // Initialize API service
        apiService = new ApiService();

        // Setup hardware security mock
        server.use(
            rest.post(`${apiConfig.baseUrl}/auth/hardware`, (req, res, ctx) => {
                return res(ctx.json(mockHardwareToken));
            })
        );

        // Configure network conditions
        server.setNetworkProfile({
            latency: 45,
            jitter: 5,
            packetLoss: 0.01,
            bandwidth: 1000000
        });

        // Wait for service initialization
        await new Promise(resolve => setTimeout(resolve, 100));
    });

    afterAll(async () => {
        // Clean up resources
        await apiService.dispose();
        server.close();
        server.resetHandlers();
    });

    describe('Hardware Security Integration', () => {
        test('should validate hardware token successfully', async () => {
            const result = await apiService.validateHardwareToken();
            expect(result).toBeTruthy();
            expect(result.deviceId).toBe(mockHardwareToken.deviceId);
            expect(result.capabilities).toEqual(mockHardwareToken.capabilities);
        });

        test('should reject invalid hardware tokens', async () => {
            server.use(
                rest.post(`${apiConfig.baseUrl}/auth/hardware`, (req, res, ctx) => {
                    return res(ctx.status(401));
                })
            );

            await expect(apiService.validateHardwareToken()).rejects.toThrow();
        });
    });

    describe('HTTP Request Handling', () => {
        test('should handle requests with retry policy', async () => {
            const response = await apiService.request({
                url: '/test',
                method: 'GET'
            });
            expect(response).toBeDefined();
        });

        test('should enforce rate limits', async () => {
            const promises = Array(101).fill(0).map(() => 
                apiService.request({ url: '/test', method: 'GET' })
            );
            await expect(Promise.all(promises)).rejects.toThrow('Rate limit exceeded');
        });

        test('should handle circuit breaker activation', async () => {
            server.use(
                rest.get(`${apiConfig.baseUrl}/test`, (req, res, ctx) => {
                    return res(ctx.status(500));
                })
            );

            const promises = Array(11).fill(0).map(() => 
                apiService.request({ url: '/test', method: 'GET' })
            );
            await expect(Promise.all(promises)).rejects.toThrow('Circuit breaker is open');
        });
    });

    describe('WebSocket Communication', () => {
        test('should establish WebSocket connection', async () => {
            const connected = await new Promise(resolve => {
                apiService['socket']?.once('connect', () => resolve(true));
            });
            expect(connected).toBe(true);
        });

        test('should handle WebSocket message compression', async () => {
            const data = { test: 'data'.repeat(1000) };
            await apiService.emit('test', data);
            const metrics = apiService.getMetrics();
            expect(metrics.wsEmit.avg).toBeLessThan(50);
        });

        test('should maintain connection pool', async () => {
            for (let i = 0; i < 4; i++) {
                await apiService.emit(`test${i}`, { data: i });
            }
            expect(apiService['connectionPool'].size).toBeLessThanOrEqual(4);
        });
    });

    describe('Fleet Management', () => {
        test('should join fleet with 32 device limit', async () => {
            const response = await apiService.request({
                url: apiConfig.endpoints.FLEET.JOIN,
                method: 'POST',
                data: { fleetId: mockFleetResponse.fleetId }
            });
            expect(response.status).toBe(FleetStatus.ACTIVE);
            expect(response.members.length).toBeLessThanOrEqual(32);
        });

        test('should handle CRDT state synchronization', async () => {
            await apiService.emit('fleet:sync', mockGameState);
            const metrics = apiService.getMetrics();
            expect(metrics.wsEmit.p95).toBeLessThan(50);
        });

        test('should manage fleet leader election', async () => {
            const response = await apiService.request({
                url: apiConfig.endpoints.FLEET.STATUS,
                method: 'GET'
            });
            expect(response.role).toBe(FleetRole.LEADER);
        });
    });

    describe('Performance Monitoring', () => {
        test('should track network latency', async () => {
            const latencyData = await apiService.monitorLatency();
            expect(latencyData.average).toBeLessThan(50);
            expect(latencyData.p95).toBeLessThan(100);
        });

        test('should collect performance metrics', () => {
            const metrics = apiService.getMetrics();
            expect(metrics).toHaveProperty('apiResponse');
            expect(metrics).toHaveProperty('wsEmit');
            expect(metrics.apiResponse.avg).toBeLessThan(50);
        });

        test('should handle backpressure', async () => {
            const largeData = new Array(1000).fill(mockLidarData);
            await apiService.emit('lidar:batch', largeData);
            const metrics = apiService.getMetrics();
            expect(metrics.wsEmit.p95).toBeLessThan(100);
        });
    });
});