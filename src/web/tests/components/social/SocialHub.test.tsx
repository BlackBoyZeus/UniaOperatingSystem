import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, test, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { WebRTC } from 'webrtc-mock';
import { PerformanceMonitor } from '@testing-library/performance';

import { SocialHub } from '../../../src/components/social/SocialHub';
import { server } from '../../mocks/server';
import { FleetStatus } from '../../../src/types/fleet.types';
import { UserStatusType } from '../../../src/interfaces/user.interface';

// Mock WebRTC service
vi.mock('../../../src/services/webrtc.service', () => ({
  WebRTCService: vi.fn().mockImplementation(() => ({
    connectToFleet: vi.fn(),
    getNetworkStats: vi.fn().mockResolvedValue({
      averageLatency: 45,
      maxLatency: 60,
      minLatency: 30,
      packetsLost: 0,
      bandwidth: 1000000,
      connectedPeers: 5,
      syncLatency: 45
    }),
    disconnectFromFleet: vi.fn()
  }))
}));

// Mock performance monitoring
const performanceMonitor = new PerformanceMonitor();

describe('SocialHub Component', () => {
  const mockProps = {
    fleetId: 'test-fleet-123',
    securityContext: 'test-context',
    privacySettings: {
      shareLocation: true,
      shareScanData: true,
      dataRetentionDays: 30,
      gdprConsent: true
    },
    region: 'na',
    powerMode: 'balanced' as const
  };

  beforeAll(() => {
    server.listen();
    // Initialize WebRTC mock
    global.RTCPeerConnection = WebRTC.RTCPeerConnection;
    global.RTCSessionDescription = WebRTC.RTCSessionDescription;
    global.RTCIceCandidate = WebRTC.RTCIceCandidate;
  });

  afterAll(() => {
    server.close();
    performanceMonitor.cleanup();
  });

  beforeEach(() => {
    server.resetHandlers();
    performanceMonitor.reset();
  });

  test('renders social hub with all required components', () => {
    render(<SocialHub {...mockProps} />);
    
    expect(screen.getByText(/Nearby Players/i)).toBeInTheDocument();
    expect(screen.getByText(/Fleet Status/i)).toBeInTheDocument();
    expect(screen.getByText(/Friends/i)).toBeInTheDocument();
  });

  test('validates fleet size limit of 32 concurrent devices', async () => {
    const { rerender } = render(<SocialHub {...mockProps} />);
    
    // Simulate adding players until limit
    for (let i = 0; i < 33; i++) {
      const player = {
        userId: `player-${i}`,
        displayName: `Player ${i}`,
        status: UserStatusType.ONLINE
      };
      
      await userEvent.click(screen.getByTestId(`player-card-${player.userId}`));
    }

    const networkStats = screen.getByText(/Players: 32\/32/i);
    expect(networkStats).toBeInTheDocument();
    
    // Attempt to add one more player
    const extraPlayer = {
      userId: 'extra-player',
      displayName: 'Extra Player',
      status: UserStatusType.ONLINE
    };
    
    await userEvent.click(screen.getByTestId(`player-card-${extraPlayer.userId}`));
    expect(screen.getByText(/Maximum fleet size reached/i)).toBeInTheDocument();
  });

  test('meets performance requirement of <16ms input latency', async () => {
    performanceMonitor.start();
    
    render(<SocialHub {...mockProps} />);
    
    // Measure interaction latency
    const button = screen.getByText(/Nearby Players/i);
    const latencies: number[] = [];
    
    for (let i = 0; i < 10; i++) {
      const start = performance.now();
      await userEvent.click(button);
      latencies.push(performance.now() - start);
    }
    
    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
    expect(avgLatency).toBeLessThan(16);
    
    performanceMonitor.stop();
  });

  test('validates P2P network latency requirement of ≤50ms', async () => {
    render(<SocialHub {...mockProps} />);
    
    // Wait for network stats update
    await waitFor(() => {
      const latencyText = screen.getByText(/Latency: (\d+)ms/);
      const latency = parseInt(latencyText.textContent!.match(/\d+/)![0]);
      expect(latency).toBeLessThanOrEqual(50);
    });
  });

  test('enforces GDPR compliance in social interactions', async () => {
    // Test with GDPR consent disabled
    const noConsentProps = {
      ...mockProps,
      privacySettings: {
        ...mockProps.privacySettings,
        gdprConsent: false
      }
    };
    
    render(<SocialHub {...noConsentProps} />);
    
    // Attempt to interact with nearby players
    const playerCard = screen.getByTestId('player-card-test');
    await userEvent.click(playerCard);
    
    expect(screen.getByText(/GDPR consent required/i)).toBeInTheDocument();
  });

  test('handles fleet formation and management', async () => {
    render(<SocialHub {...mockProps} />);
    
    // Create fleet
    await userEvent.click(screen.getByText(/Create Fleet/i));
    
    await waitFor(() => {
      expect(screen.getByText(/Fleet Status: ACTIVE/i)).toBeInTheDocument();
    });
    
    // Add members
    const player = {
      userId: 'test-player',
      displayName: 'Test Player',
      status: UserStatusType.ONLINE
    };
    
    await userEvent.click(screen.getByTestId(`player-card-${player.userId}`));
    
    expect(screen.getByText(/Players: 2\/32/i)).toBeInTheDocument();
  });

  test('adapts to power-save mode', async () => {
    const powerSaveProps = {
      ...mockProps,
      powerMode: 'powersave' as const
    };
    
    render(<SocialHub {...powerSaveProps} />);
    
    // Verify reduced update frequency
    const initialStats = screen.getByText(/Latency: \d+ms/);
    const initialValue = initialStats.textContent;
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const updatedStats = screen.getByText(/Latency: \d+ms/);
    expect(updatedStats.textContent).toBe(initialValue);
  });

  test('handles connection quality degradation', async () => {
    render(<SocialHub {...mockProps} />);
    
    // Simulate poor network conditions
    server.use(
      rest.get('/fleet/status', (req, res, ctx) => {
        return res(
          ctx.json({
            averageLatency: 100,
            packetsLost: 10,
            bandwidth: 500000
          })
        );
      })
    );
    
    await waitFor(() => {
      expect(screen.getByText(/Fleet Status: DEGRADED/i)).toBeInTheDocument();
    });
  });

  test('maintains security context during fleet operations', async () => {
    const { container } = render(<SocialHub {...mockProps} />);
    
    // Verify security attributes
    expect(container.querySelector('[data-security-context]'))
      .toHaveAttribute('data-security-context', mockProps.securityContext);
    
    // Attempt fleet operation
    await userEvent.click(screen.getByText(/Create Fleet/i));
    
    // Verify security context maintained
    expect(container.querySelector('[data-security-context]'))
      .toHaveAttribute('data-security-context', mockProps.securityContext);
  });
});