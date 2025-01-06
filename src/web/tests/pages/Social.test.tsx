import React from 'react';
import { render, screen, fireEvent, waitFor, within, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import WebRTCMock from '@testing-library/webrtc-mock';
import { LiDAR } from '@lidar/testing';

import Social, { SocialPageProps } from '../../src/pages/Social';
import { useFleet, FleetContext } from '../../src/hooks/useFleet';

// Mock dependencies
vi.mock('../../src/hooks/useFleet');
vi.mock('@lidar/testing');
vi.mock('@testing-library/webrtc-mock');

// Test data
const mockFleet = {
  id: 'fleet-123',
  name: 'Test Fleet',
  maxDevices: 32,
  currentDevices: 5,
  leader: 'device-1',
  members: ['device-1', 'device-2', 'device-3', 'device-4', 'device-5'],
  createdAt: new Date()
};

const mockNetworkStats = {
  latency: 45,
  bandwidth: 1000000,
  peers: 5,
  syncState: 'synchronized'
};

const mockLiDARData = {
  scanId: 'scan-123',
  pointCloud: new Float32Array(1000),
  timestamp: Date.now(),
  nearbyPlayers: [
    { id: 'player-1', distance: 2.5, quality: 0.95 },
    { id: 'player-2', distance: 3.8, quality: 0.85 }
  ]
};

describe('Social Page', () => {
  let mockWebRTC: WebRTCMock;
  let mockLiDARScanner: LiDAR;

  beforeEach(() => {
    mockWebRTC = new WebRTCMock({
      latency: 45,
      bandwidth: 1000000,
      maxConnections: 32
    });

    mockLiDARScanner = new LiDAR({
      resolution: 0.01,
      range: 5,
      frequency: 30
    });

    // Mock fleet context
    (useFleet as jest.Mock).mockReturnValue({
      currentFleet: mockFleet,
      fleetMembers: new Map(mockFleet.members.map(id => [id, { id }])),
      networkStats: mockNetworkStats,
      createFleet: vi.fn(),
      joinFleet: vi.fn(),
      leaveFleet: vi.fn()
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    mockWebRTC.cleanup();
    mockLiDARScanner.cleanup();
  });

  it('should render social page with fleet status', async () => {
    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    expect(screen.getByText(/Fleet Status/i)).toBeInTheDocument();
    expect(screen.getByText(/5\/32/i)).toBeInTheDocument();
    expect(screen.getByText(/45ms/i)).toBeInTheDocument();
  });

  it('should validate hardware token before joining fleet', async () => {
    const joinFleet = vi.fn();
    (useFleet as jest.Mock).mockReturnValue({
      ...useFleet(),
      joinFleet
    });

    render(
      <Social
        hardwareToken="invalid-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    const joinButton = screen.getByRole('button', { name: /join fleet/i });
    await fireEvent.click(joinButton);

    expect(joinFleet).not.toHaveBeenCalled();
    expect(screen.getByText(/hardware security validation failed/i)).toBeInTheDocument();
  });

  it('should enforce fleet size limit of 32 devices', async () => {
    const mockFullFleet = {
      ...mockFleet,
      currentDevices: 32,
      members: Array.from({ length: 32 }, (_, i) => `device-${i}`)
    };

    (useFleet as jest.Mock).mockReturnValue({
      ...useFleet(),
      currentFleet: mockFullFleet
    });

    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    const joinButton = screen.getByRole('button', { name: /join fleet/i });
    expect(joinButton).toBeDisabled();
  });

  it('should monitor and display network quality metrics', async () => {
    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    await mockWebRTC.simulateNetworkChange({
      latency: 60,
      bandwidth: 800000
    });

    await waitFor(() => {
      expect(screen.getByText(/60ms/i)).toBeInTheDocument();
      expect(screen.getByText(/0.8 Mbps/i)).toBeInTheDocument();
    });
  });

  it('should integrate LiDAR scanning for proximity detection', async () => {
    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    await act(async () => {
      await mockLiDARScanner.scan();
      await mockLiDARScanner.processScan(mockLiDARData);
    });

    expect(screen.getByText(/2.5m away/i)).toBeInTheDocument();
    expect(screen.getByText(/3.8m away/i)).toBeInTheDocument();
  });

  it('should adapt UI performance based on power mode', async () => {
    const { rerender } = render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    const container = screen.getByTestId('social-layout');
    expect(getComputedStyle(container).getPropertyValue('--animation-duration')).toBe('300ms');

    rerender(
      <Social
        hardwareToken="mock-token"
        powerMode="powersave"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    expect(getComputedStyle(container).getPropertyValue('--animation-duration')).toBe('500ms');
  });

  it('should handle fleet state synchronization', async () => {
    const syncFleetState = vi.fn();
    (useFleet as jest.Mock).mockReturnValue({
      ...useFleet(),
      syncFleetState
    });

    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 45, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.95 }}
      />
    );

    await act(async () => {
      await mockWebRTC.simulateStateSync({
        type: 'game_state',
        payload: { timestamp: Date.now() }
      });
    });

    expect(syncFleetState).toHaveBeenCalled();
  });

  it('should enforce network latency requirements', async () => {
    render(
      <Social
        hardwareToken="mock-token"
        powerMode="performance"
        networkQuality={{ latency: 75, bandwidth: 1000000, packetsLost: 0, connectionScore: 0.7 }}
      />
    );

    const joinButton = screen.getByRole('button', { name: /join fleet/i });
    await fireEvent.click(joinButton);

    expect(screen.getByText(/network quality insufficient/i)).toBeInTheDocument();
  });
});