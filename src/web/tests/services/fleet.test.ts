import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals'; // @version ^29.0.0
import * as Automerge from 'automerge'; // @version ^2.0.0

import FleetService from '../../src/services/fleet.service';
import WebRTCService from '../../src/services/webrtc.service';
import ApiService from '../../src/services/api.service';
import { 
    FleetStatus, 
    FleetRole, 
    FleetMessageType,
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD 
} from '../../src/types/fleet.types';

// Mock services
jest.mock('../../src/services/webrtc.service');
jest.mock('../../src/services/api.service');

describe('FleetService', () => {
    let fleetService: FleetService;
    let webrtcService: jest.Mocked<WebRTCService>;
    let apiService: jest.Mocked<ApiService>;

    beforeEach(() => {
        webrtcService = new WebRTCService() as jest.Mocked<WebRTCService>;
        apiService = new ApiService() as jest.Mocked<ApiService>;
        fleetService = new FleetService(webrtcService, apiService);
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('Fleet Creation', () => {
        test('should create fleet with valid parameters', async () => {
            const fleetName = 'TestFleet';
            const maxDevices = 32;

            apiService.request.mockResolvedValueOnce({
                id: '123',
                name: fleetName,
                maxDevices,
                members: [],
                status: FleetStatus.ACTIVE,
                networkStats: {
                    averageLatency: 0,
                    maxLatency: 0,
                    minLatency: Number.MAX_VALUE,
                    packetsLost: 0,
                    bandwidth: 0,
                    connectedPeers: 0,
                    syncLatency: 0
                }
            });

            const fleet = await fleetService.createFleet(fleetName, maxDevices);

            expect(fleet).toBeDefined();
            expect(fleet.name).toBe(fleetName);
            expect(fleet.maxDevices).toBe(maxDevices);
            expect(apiService.request).toHaveBeenCalledWith({
                url: '/fleet/create',
                method: 'POST',
                data: { name: fleetName, maxDevices }
            });
        });

        test('should enforce 32-device fleet limit', async () => {
            await expect(fleetService.createFleet('TestFleet', 33))
                .rejects
                .toThrow(`Fleet size cannot exceed ${MAX_FLEET_SIZE} devices`);
        });

        test('should initialize WebRTC connections for fleet members', async () => {
            const fleetName = 'TestFleet';
            const members = [
                { id: '1', deviceId: 'device1', role: FleetRole.LEADER },
                { id: '2', deviceId: 'device2', role: FleetRole.MEMBER }
            ];

            apiService.request.mockResolvedValueOnce({
                id: '123',
                name: fleetName,
                maxDevices: 32,
                members,
                status: FleetStatus.ACTIVE,
                networkStats: { averageLatency: 0 }
            });

            await fleetService.createFleet(fleetName);

            expect(webrtcService.initializeConnection).toHaveBeenCalledTimes(members.length);
        });
    });

    describe('Fleet State Management', () => {
        test('should synchronize fleet state using CRDT', async () => {
            const mockState = Automerge.init();
            const mockPeerState = Automerge.change(mockState, 'test', doc => {
                doc.test = 'value';
            });

            webrtcService.sendGameState.mockResolvedValueOnce(mockPeerState);

            await fleetService.syncFleetState();

            expect(webrtcService.sendGameState).toHaveBeenCalled();
        });

        test('should handle network quality degradation', async () => {
            webrtcService.monitorNetworkQuality.mockResolvedValueOnce({
                connectionQuality: 0.5,
                latency: 75
            });

            await fleetService['monitorFleetHealth']();

            expect(webrtcService.monitorNetworkQuality).toHaveBeenCalled();
        });

        test('should maintain state consistency across fleet', async () => {
            const mockConnections = new Map([
                ['peer1', { peerConnection: {}, dataChannel: {} }],
                ['peer2', { peerConnection: {}, dataChannel: {} }]
            ]);

            Object.defineProperty(fleetService, 'connections', {
                value: mockConnections
            });

            await fleetService.syncFleetState();

            expect(webrtcService.sendGameState).toHaveBeenCalledTimes(mockConnections.size);
        });
    });

    describe('Leader Election and Failover', () => {
        test('should handle leader failover gracefully', async () => {
            const mockConnections = new Map([
                ['peer1', { peerConnection: {}, dataChannel: {} }],
                ['peer2', { peerConnection: {}, dataChannel: {} }]
            ]);

            Object.defineProperty(fleetService, 'connections', {
                value: mockConnections
            });

            await fleetService['handleLeaderFailover']();

            expect(webrtcService.sendGameState).toHaveBeenCalled();
        });

        test('should select new leader based on network metrics', async () => {
            const mockMetrics = new Map([
                ['peer1', 0.9],
                ['peer2', 0.7]
            ]);

            Object.defineProperty(fleetService, 'networkMetrics', {
                value: mockMetrics
            });

            await fleetService['handleLeaderFailover']();

            const newLeader = fleetService['selectNewLeader']();
            expect(newLeader).toBe('peer1');
        });
    });

    describe('Network Quality Monitoring', () => {
        test('should monitor network quality for all peers', async () => {
            const mockConnections = new Map([
                ['peer1', { peerConnection: {} }],
                ['peer2', { peerConnection: {} }]
            ]);

            Object.defineProperty(fleetService, 'connections', {
                value: mockConnections
            });

            webrtcService.monitorNetworkQuality.mockResolvedValue({
                connectionQuality: 0.9,
                latency: 45
            });

            await fleetService['monitorFleetHealth']();

            expect(webrtcService.monitorNetworkQuality).toHaveBeenCalledTimes(mockConnections.size);
        });

        test('should handle degraded connections', async () => {
            const peerId = 'peer1';
            const mockConnections = new Map([
                [peerId, { peerConnection: {} }]
            ]);

            Object.defineProperty(fleetService, 'connections', {
                value: mockConnections
            });

            webrtcService.monitorNetworkQuality.mockResolvedValueOnce({
                connectionQuality: 0.5,
                latency: 75
            });

            await fleetService['monitorFleetHealth']();

            expect(fleetService['handleDegradedConnection']).toHaveBeenCalledWith(peerId);
        });

        test('should enforce latency thresholds', async () => {
            webrtcService.monitorNetworkQuality.mockResolvedValueOnce({
                connectionQuality: 0.9,
                latency: MAX_LATENCY_THRESHOLD + 10
            });

            const mockUpdateMetrics = jest.spyOn(fleetService as any, 'updateNetworkMetrics');

            await fleetService['monitorFleetHealth']();

            expect(mockUpdateMetrics).toHaveBeenCalled();
            expect(fleetService['networkMetrics'].get('latency')).toBeLessThanOrEqual(MAX_LATENCY_THRESHOLD);
        });
    });

    describe('Fleet Cleanup', () => {
        test('should clean up resources on leave', async () => {
            const mockStopMonitoring = jest.spyOn(fleetService as any, 'stopMonitoring');
            const mockCleanupConnections = jest.spyOn(fleetService as any, 'cleanupConnections');

            await fleetService.leaveFleet();

            expect(mockStopMonitoring).toHaveBeenCalled();
            expect(mockCleanupConnections).toHaveBeenCalled();
            expect(apiService.request).toHaveBeenCalledWith({
                url: '/fleet/leave',
                method: 'POST',
                data: expect.any(Object)
            });
        });

        test('should handle cleanup errors gracefully', async () => {
            apiService.request.mockRejectedValueOnce(new Error('Cleanup failed'));

            await expect(fleetService.leaveFleet()).rejects.toThrow('Cleanup failed');
        });
    });
});