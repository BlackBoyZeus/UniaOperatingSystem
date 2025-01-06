import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { performance } from 'perf_hooks';
import { mock } from 'mock-webrtc';

// Internal imports
import Fleet from '../../src/pages/Fleet';
import { server, handlers } from '../mocks/server';

// Mock WebRTC for P2P testing
mock.setup();

// Helper function to render with required providers and monitor performance
const renderWithProviders = (ui: React.ReactElement, options = {}) => {
  const startTime = performance.now();
  const result = render(ui);
  const renderTime = performance.now() - startTime;

  return {
    ...result,
    renderTime,
    getPerformanceMetrics: () => ({
      renderTime,
      gpuAccelerated: document.querySelector('[style*="transform"]') !== null,
      animationFrameTime: performance.now() - startTime
    })
  };
};

describe('Fleet Page Component', () => {
  beforeEach(() => {
    server.listen();
  });

  afterEach(() => {
    server.resetHandlers();
    server.close();
  });

  it('renders fleet management interface with correct performance', async () => {
    const { renderTime, getPerformanceMetrics } = renderWithProviders(<Fleet />);

    // Verify render performance meets 16ms target
    expect(renderTime).toBeLessThan(16);
    expect(getPerformanceMetrics().gpuAccelerated).toBe(true);

    // Verify core UI elements
    expect(screen.getByText('Fleet Management')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Create Fleet/i })).toBeEnabled();
  });

  it('enforces 32-device fleet size limit with proper validation', async () => {
    renderWithProviders(<Fleet />);

    // Create fleet with maximum size
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Fleet Name/i), 'Test Fleet');
    await userEvent.type(screen.getByLabelText(/Maximum Devices/i), '32');
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Verify fleet creation success
    await waitFor(() => {
      expect(screen.getByText('Test Fleet')).toBeInTheDocument();
      expect(screen.getByText('0/32')).toBeInTheDocument();
    });

    // Attempt to exceed limit
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Maximum Devices/i), '33');

    // Verify validation error
    expect(screen.getByText(/Fleet size cannot exceed 32 devices/i)).toBeInTheDocument();
  });

  it('maintains P2P network performance within specifications', async () => {
    renderWithProviders(<Fleet />);

    // Create and join fleet
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Fleet Name/i), 'P2P Test Fleet');
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Verify network metrics
    await waitFor(() => {
      const networkStats = screen.getByText(/45ms/);
      expect(networkStats).toBeInTheDocument();
      expect(parseInt(networkStats.textContent!)).toBeLessThanOrEqual(50);
    });

    // Test WebRTC connection establishment
    const peerConnections = mock.getPeerConnections();
    expect(peerConnections.length).toBeGreaterThan(0);
    expect(peerConnections[0].connectionState).toBe('connected');
  });

  it('handles fleet member synchronization with CRDT', async () => {
    renderWithProviders(<Fleet />);

    // Create fleet and add members
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Fleet Name/i), 'Sync Test Fleet');
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Simulate member joins
    server.use(
      handlers.createFleetHandler,
      handlers.joinFleetHandler,
      handlers.webRTCHandler,
      handlers.crdtSyncHandler
    );

    // Verify state synchronization
    await waitFor(() => {
      const memberList = screen.getByRole('list', { name: /Fleet Members/i });
      const members = within(memberList).getAllByRole('listitem');
      expect(members).toHaveLength(2);
      expect(screen.getByText(/Sync: 100%/i)).toBeInTheDocument();
    });
  });

  it('implements proper fleet leader election and failover', async () => {
    renderWithProviders(<Fleet />);

    // Create fleet with leader
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Fleet Name/i), 'Leader Test Fleet');
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Verify initial leader assignment
    await waitFor(() => {
      expect(screen.getByText(/Leader/i)).toBeInTheDocument();
    });

    // Simulate leader disconnection
    server.use(handlers.leaderFailoverHandler);

    // Verify backup leader promotion
    await waitFor(() => {
      const newLeader = screen.getByText(/New Leader/i);
      expect(newLeader).toBeInTheDocument();
      expect(screen.getByText(/Failover completed/i)).toBeInTheDocument();
    });
  });

  it('optimizes UI performance with power-aware animations', async () => {
    const { getPerformanceMetrics } = renderWithProviders(<Fleet />);

    // Create fleet with animations
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Verify animation performance
    const metrics = getPerformanceMetrics();
    expect(metrics.animationFrameTime).toBeLessThan(16);
    expect(document.querySelector('[style*="will-change"]')).not.toBeNull();
  });

  it('handles network degradation gracefully', async () => {
    renderWithProviders(<Fleet />);

    // Create fleet
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));
    await userEvent.type(screen.getByLabelText(/Fleet Name/i), 'Network Test Fleet');
    fireEvent.click(screen.getByRole('button', { name: /Create Fleet/i }));

    // Simulate network degradation
    server.use(handlers.networkDegradationHandler);

    // Verify degradation handling
    await waitFor(() => {
      expect(screen.getByText(/Network Quality: Poor/i)).toBeInTheDocument();
      expect(screen.getByText(/Attempting reconnection/i)).toBeInTheDocument();
    });

    // Verify recovery
    await waitFor(() => {
      expect(screen.getByText(/Network Quality: Excellent/i)).toBeInTheDocument();
    }, { timeout: 5000 });
  });
});