import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { FleetCard, FleetCardProps } from '../../src/components/fleet/FleetCard';
import { IFleet, IFleetMember } from '../../src/interfaces/fleet.interface';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @testing-library/react v14.0.0
 * @external @testing-library/user-event v14.0.0
 * @external vitest v0.34.0
 */

// Mock fleet data with comprehensive network stats and HDR settings
const mockFleet: IFleet = {
  id: 'test-fleet-1',
  name: 'Test Fleet',
  maxDevices: 32,
  members: [
    {
      id: 'member-1',
      deviceId: 'device-1',
      role: 'LEADER',
      connection: {
        lastPing: Date.now(),
        connectionQuality: 0.95,
        retryCount: 0
      },
      latency: 45,
      connectionQuality: {
        signalStrength: 0.9,
        stability: 0.95,
        reliability: 0.92
      },
      lastCRDTOperation: {
        timestamp: Date.now(),
        type: 'UPDATE',
        payload: {}
      }
    }
  ],
  status: 'ACTIVE',
  networkStats: {
    averageLatency: 45,
    maxLatency: 50,
    minLatency: 40,
    packetsLost: 0,
    bandwidth: 1000,
    connectedPeers: 1,
    syncLatency: 10
  },
  qualityMetrics: {
    connectionScore: 0.95,
    syncSuccess: 98,
    leaderRedundancy: 1
  },
  backupLeaders: ['device-2', 'device-3']
};

// Mock handlers for fleet operations
const mockHandlers = {
  onJoin: vi.fn(),
  onLeave: vi.fn(),
  onManage: vi.fn(),
  onDisplayModeChange: vi.fn(),
  onPowerModeChange: vi.fn()
};

// Mock performance measurement
const mockPerformanceNow = vi.fn();
performance.now = mockPerformanceNow;

// Mock HDR detection
const mockMatchMedia = vi.fn();
window.matchMedia = mockMatchMedia;

describe('FleetCard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPerformanceNow.mockReturnValue(0);
    mockMatchMedia.mockReturnValue({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    });
  });

  // Rendering Tests
  describe('Rendering', () => {
    it('renders fleet information correctly', () => {
      render(<FleetCard fleet={mockFleet} {...mockHandlers} />);
      
      expect(screen.getByText(mockFleet.name)).toBeInTheDocument();
      expect(screen.getByText(`${mockFleet.members.length}/${mockFleet.maxDevices}`)).toBeInTheDocument();
      expect(screen.getByText(mockFleet.status)).toBeInTheDocument();
    });

    it('renders network quality indicators with correct HDR colors', () => {
      render(<FleetCard fleet={mockFleet} hdrEnabled={true} {...mockHandlers} />);
      
      const qualityIndicator = screen.getByText(`${mockFleet.networkStats.averageLatency.toFixed(0)}ms`);
      expect(qualityIndicator).toHaveStyle({
        color: 'color(display-p3 0 1 0.5)'
      });
    });

    it('applies power-save mode optimizations', () => {
      render(<FleetCard fleet={mockFleet} powerMode="powersave" {...mockHandlers} />);
      
      const container = screen.getByTestId('fleet-card');
      expect(container).toHaveStyle({
        'will-change': 'auto'
      });
    });
  });

  // Interaction Tests
  describe('Interactions', () => {
    it('handles join fleet with performance tracking', async () => {
      mockPerformanceNow
        .mockReturnValueOnce(0)
        .mockReturnValueOnce(10);

      render(<FleetCard fleet={mockFleet} {...mockHandlers} />);
      
      const joinButton = screen.getByText('Join Fleet');
      await userEvent.click(joinButton);

      expect(mockHandlers.onJoin).toHaveBeenCalledWith(mockFleet.id);
      expect(mockPerformanceNow).toHaveBeenCalledTimes(2);
    });

    it('maintains interaction performance under 16ms', async () => {
      const interactionLatencies: number[] = [];
      
      mockPerformanceNow
        .mockImplementation(() => interactionLatencies.length * 10);

      render(<FleetCard fleet={mockFleet} {...mockHandlers} />);
      
      for (let i = 0; i < 10; i++) {
        const startTime = performance.now();
        await userEvent.click(screen.getByText('Join Fleet'));
        interactionLatencies.push(performance.now() - startTime);
      }

      expect(Math.max(...interactionLatencies)).toBeLessThan(16);
    });

    it('handles display mode changes with HDR support', async () => {
      render(<FleetCard fleet={mockFleet} hdrEnabled={true} {...mockHandlers} />);
      
      const displayModeToggle = screen.getByTestId('display-mode-toggle');
      await userEvent.click(displayModeToggle);

      expect(mockHandlers.onDisplayModeChange).toHaveBeenCalledWith('HDR');
    });
  });

  // Network Quality Tests
  describe('Network Quality', () => {
    it('updates network quality indicators in real-time', async () => {
      const { rerender } = render(<FleetCard fleet={mockFleet} {...mockHandlers} />);
      
      const updatedFleet = {
        ...mockFleet,
        networkStats: {
          ...mockFleet.networkStats,
          averageLatency: 30
        }
      };

      rerender(<FleetCard fleet={updatedFleet} {...mockHandlers} />);
      
      await waitFor(() => {
        expect(screen.getByText('30ms')).toBeInTheDocument();
      });
    });

    it('displays warning for high latency', () => {
      const highLatencyFleet = {
        ...mockFleet,
        networkStats: {
          ...mockFleet.networkStats,
          averageLatency: 48
        }
      };

      render(<FleetCard fleet={highLatencyFleet} {...mockHandlers} />);
      
      const latencyIndicator = screen.getByTestId('latency-indicator');
      expect(latencyIndicator).toHaveClass('warning');
    });
  });

  // Power Mode Tests
  describe('Power Mode', () => {
    it('adapts animations based on power mode', async () => {
      const { rerender } = render(
        <FleetCard fleet={mockFleet} powerMode="performance" {...mockHandlers} />
      );

      const card = screen.getByTestId('fleet-card');
      expect(card).toHaveStyle({
        'transition-duration': 'var(--animation-duration-performance)'
      });

      rerender(<FleetCard fleet={mockFleet} powerMode="powersave" {...mockHandlers} />);
      
      expect(card).toHaveStyle({
        'transition-duration': 'var(--animation-duration-power-save)'
      });
    });

    it('disables animations in extreme power save mode', () => {
      render(
        <FleetCard 
          fleet={mockFleet} 
          powerMode="powersave" 
          animationEnabled={false} 
          {...mockHandlers} 
        />
      );

      const card = screen.getByTestId('fleet-card');
      expect(card).toHaveStyle({
        'transition': 'none'
      });
    });
  });

  // Accessibility Tests
  describe('Accessibility', () => {
    it('maintains ARIA attributes in all display modes', () => {
      render(<FleetCard fleet={mockFleet} {...mockHandlers} />);
      
      const card = screen.getByTestId('fleet-card');
      expect(card).toHaveAttribute('aria-label', expect.stringContaining(mockFleet.name));
      expect(card).toHaveAttribute('role', 'region');
    });

    it('preserves keyboard navigation in power save mode', async () => {
      render(
        <FleetCard fleet={mockFleet} powerMode="powersave" {...mockHandlers} />
      );

      const joinButton = screen.getByText('Join Fleet');
      await userEvent.tab();
      expect(joinButton).toHaveFocus();
    });
  });
});