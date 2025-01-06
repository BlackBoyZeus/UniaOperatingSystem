import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals'; // v29.0.0
import { mockWebRTC } from 'mock-webrtc'; // v1.0.0

import { FleetState } from '../../src/core/fleet/FleetState';
import { FleetManager } from '../../src/core/fleet/FleetManager';
import { 
  FleetRole, 
  FleetStatus, 
  MeshTopologyType,
  FleetAuthMethod,
  JoinPolicy 
} from '../../interfaces/fleet.interface';

// Mock configurations and test data
const mockFleetConfig = {
  id: 'test-fleet-1',
  name: 'Test Fleet',
  maxDevices: 32,
  members: [],
  meshConfig: {
    topology: MeshTopologyType.HYBRID,
    maxPeers: 32,
    reconnectStrategy: {
      maxAttempts: 3,
      backoffMultiplier: 1.5,
      initialDelay: 1000,
      maxDelay: 5000
    },
    peerTimeout: 5000,
    signalServer: 'wss://signal.tald.unia/fleet',
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    meshQuality: {
      connectionDensity: 1.0,
      redundancyFactor: 0.5,
      meshStability: 1.0,
      routingEfficiency: 1.0
    }
  },
  networkStats: {
    averageLatency: 0,
    peakLatency: 0,
    packetLoss: 0,
    bandwidth: {
      current: 0,
      peak: 0,
      average: 0,
      totalTransferred: 0,
      lastMeasured: Date.now()
    },
    connectionQuality: 1.0,
    meshHealth: 1.0,
    lastUpdate: Date.now()
  },
  securityConfig: {
    encryptionEnabled: true,
    authenticationMethod: FleetAuthMethod.TOKEN,
    accessControl: {
      allowedDevices: [],
      bannedDevices: [],
      joinPolicy: JoinPolicy.OPEN,
      rolePermissions: new Map()
    }
  },
  createdAt: Date.now(),
  lastUpdated: Date.now()
};

const mockMember = {
  id: 'test-member-1',
  deviceId: 'test-device-1',
  role: FleetRole.MEMBER,
  status: FleetStatus.ACTIVE,
  joinedAt: Date.now(),
  lastActive: Date.now(),
  position: { x: 0, y: 0, z: 0, timestamp: Date.now(), accuracy: 1.0 },
  capabilities: {
    lidarSupport: true,
    maxRange: 5.0,
    processingPower: 1000,
    networkBandwidth: 2000,
    batteryLevel: 100
  }
};

describe('FleetState', () => {
  let fleetState: FleetState;

  beforeEach(() => {
    fleetState = new FleetState(mockFleetConfig);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('should initialize with valid configuration', () => {
    expect(fleetState).toBeDefined();
    expect(fleetState.getMetrics()).toMatchObject({
      averageLatency: 0,
      syncSuccessRate: 1.0,
      memberCount: 0
    });
  });

  test('should add member with validation', async () => {
    const addMemberSpy = jest.spyOn(fleetState, 'addMember');
    await fleetState.addMember(mockMember);

    expect(addMemberSpy).toHaveBeenCalledWith(mockMember);
    expect(fleetState.getMetrics().memberCount).toBe(1);
  });

  test('should enforce fleet size limit', async () => {
    const members = Array.from({ length: 33 }, (_, i) => ({
      ...mockMember,
      id: `test-member-${i}`,
      deviceId: `test-device-${i}`
    }));

    await expect(async () => {
      for (const member of members) {
        await fleetState.addMember(member);
      }
    }).rejects.toThrow(/Fleet size limit/);
  });

  test('should validate member capabilities', async () => {
    const invalidMember = {
      ...mockMember,
      capabilities: {
        ...mockMember.capabilities,
        lidarSupport: false
      }
    };

    await expect(fleetState.addMember(invalidMember))
      .rejects.toThrow(/LiDAR support required/);
  });

  test('should synchronize state with CRDT', async () => {
    const change = {
      documentId: 'test-doc',
      operation: 'INSERT',
      timestamp: Date.now(),
      retryCount: 0
    };

    await fleetState.synchronizeState(change);
    const state = fleetState.getState();
    expect(state.stateVersion).toBeGreaterThan(0);
  });

  test('should handle network failures gracefully', async () => {
    const errorSpy = jest.spyOn(fleetState, 'handleError');
    const change = {
      documentId: 'test-doc',
      operation: 'INSERT',
      timestamp: Date.now() - 60000, // Simulate old change
      retryCount: 0
    };

    await expect(fleetState.synchronizeState(change))
      .rejects.toThrow(/latency .* exceeds threshold/);
    expect(errorSpy).toHaveBeenCalled();
  });
});

describe('FleetManager', () => {
  let fleetManager: FleetManager;
  let mockRTCPeerConnection: jest.Mock;

  beforeEach(() => {
    mockRTCPeerConnection = mockWebRTC.RTCPeerConnection;
    fleetManager = new FleetManager(mockFleetConfig, mockFleetConfig.securityConfig);
  });

  afterEach(() => {
    fleetManager.dispose();
    jest.clearAllMocks();
  });

  test('should create fleet with monitoring', async () => {
    const fleet = await fleetManager.createFleet(mockFleetConfig);
    expect(fleet.id).toBe(mockFleetConfig.id);
    expect(fleet.meshConfig.topology).toBe(MeshTopologyType.HYBRID);
  });

  test('should handle member joining with WebRTC setup', async () => {
    const joinSpy = jest.spyOn(fleetManager, 'joinFleet');
    await fleetManager.joinFleet(mockMember.id, mockMember.capabilities);

    expect(joinSpy).toHaveBeenCalled();
    expect(mockRTCPeerConnection).toHaveBeenCalled();
  });

  test('should monitor member performance', async () => {
    await fleetManager.joinFleet(mockMember.id, mockMember.capabilities);
    const metrics = fleetManager.getMetrics();

    expect(metrics.memberCount).toBe(1);
    expect(metrics.averageLatency).toBeDefined();
    expect(metrics.syncSuccessRate).toBeGreaterThan(0);
  });

  test('should handle network failures with circuit breaker', async () => {
    mockRTCPeerConnection.mockImplementation(() => {
      throw new Error('Network failure');
    });

    await expect(fleetManager.joinFleet(mockMember.id, mockMember.capabilities))
      .rejects.toThrow(/Network failure/);

    const metrics = fleetManager.getMetrics();
    expect(metrics.syncSuccessRate).toBeLessThan(1);
  });

  test('should enforce security policies', async () => {
    const secureFleetConfig = {
      ...mockFleetConfig,
      securityConfig: {
        ...mockFleetConfig.securityConfig,
        joinPolicy: JoinPolicy.INVITE_ONLY
      }
    };

    const secureManager = new FleetManager(secureFleetConfig, secureFleetConfig.securityConfig);
    await expect(secureManager.joinFleet(mockMember.id, mockMember.capabilities))
      .rejects.toThrow(/Invite required/);
  });

  test('should maintain mesh network topology', async () => {
    const members = Array.from({ length: 5 }, (_, i) => ({
      ...mockMember,
      id: `test-member-${i}`,
      deviceId: `test-device-${i}`
    }));

    for (const member of members) {
      await fleetManager.joinFleet(member.id, member.capabilities);
    }

    const metrics = fleetManager.getMetrics();
    expect(metrics.memberCount).toBe(5);
  });

  test('should handle member departure cleanly', async () => {
    await fleetManager.joinFleet(mockMember.id, mockMember.capabilities);
    await fleetManager.leaveFleet(mockMember.id);

    const metrics = fleetManager.getMetrics();
    expect(metrics.memberCount).toBe(0);
  });

  test('should validate network performance thresholds', async () => {
    const performanceSpy = jest.spyOn(fleetManager as any, 'monitorPerformance');
    await fleetManager.joinFleet(mockMember.id, mockMember.capabilities);

    expect(performanceSpy).toHaveBeenCalled();
    const metrics = fleetManager.getMetrics();
    expect(metrics.averageLatency).toBeLessThanOrEqual(50); // 50ms threshold
  });
});