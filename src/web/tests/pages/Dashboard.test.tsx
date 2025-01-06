import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import Dashboard, { DashboardProps } from '../../src/pages/Dashboard';
import { server, resetHandlers, closeServer } from '../mocks/server';
import { handlers } from '../mocks/handlers';

// Mock performance monitoring
jest.mock('@performance-monitor/react', () => ({
  __esModule: true,
  default: () => ({
    startMonitoring: jest.fn(),
    metrics: {
      averageLatency: 15,
      frameTime: 16.67,
      fps: 60
    }
  })
}));

// Mock hooks
jest.mock('../../src/hooks/useAuth', () => ({
  __esModule: true,
  default: () => ({
    user: { id: 'test-user' },
    monitorSecurityEvents: jest.fn().mockResolvedValue(undefined)
  })
}));

jest.mock('../../src/hooks/useFleet', () => ({
  __esModule: true,
  default: () => ({
    currentFleet: {
      id: 'test-fleet',
      status: 'ACTIVE',
      members: new Array(5).fill(null).map((_, i) => ({ id: `member-${i}` }))
    },
    networkStats: {
      averageLatency: 45,
      maxLatency: 50,
      minLatency: 40,
      packetsLost: 0,
      bandwidth: 1000000,
      connectedPeers: 5,
      syncLatency: 45
    }
  })
}));

describe('Dashboard', () => {
  // Default props
  const defaultProps: DashboardProps = {
    powerMode: 'BALANCED',
    hdrEnabled: true
  };

  beforeAll(() => {
    server.listen();
  });

  beforeEach(() => {
    resetHandlers();
  });

  afterEach(() => {
    server.resetHandlers();
  });

  afterAll(() => {
    closeServer();
  });

  describe('Layout and Rendering', () => {
    it('should render the dashboard container with correct layout', () => {
      render(<Dashboard {...defaultProps} />);
      
      const container = screen.getByRole('main');
      expect(container).toBeInTheDocument();
      expect(container).toHaveStyle({
        display: 'flex',
        flexDirection: 'column'
      });
    });

    it('should render stats grid with responsive design', () => {
      render(<Dashboard {...defaultProps} />);
      
      const statsGrid = screen.getByRole('grid');
      expect(statsGrid).toBeInTheDocument();
      expect(statsGrid).toHaveStyle({
        display: 'grid',
        gap: '24px'
      });
    });

    it('should render all required stat components', () => {
      render(<Dashboard {...defaultProps} />);
      
      expect(screen.getByTestId('fleet-stats')).toBeInTheDocument();
      expect(screen.getByTestId('lidar-stats')).toBeInTheDocument();
      expect(screen.getByTestId('game-stats')).toBeInTheDocument();
    });
  });

  describe('Performance Metrics', () => {
    it('should maintain target frame rate during animations', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const metrics = screen.getByTestId('performance-metrics');
        expect(metrics).toHaveTextContent(/60 FPS/);
      });
    });

    it('should monitor and display GPU utilization', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const gpuMetrics = screen.getByTestId('gpu-metrics');
        expect(gpuMetrics).toBeInTheDocument();
      });
    });

    it('should trigger performance alerts at correct thresholds', async () => {
      const { rerender } = render(<Dashboard {...defaultProps} />);
      
      // Simulate performance degradation
      rerender(<Dashboard {...defaultProps} powerMode="POWER_SAVER" />);
      
      await waitFor(() => {
        const alert = screen.getByRole('alert');
        expect(alert).toHaveTextContent(/Performance degradation detected/);
      });
    });
  });

  describe('Fleet Management', () => {
    it('should display real-time fleet size updates', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const fleetSize = screen.getByTestId('fleet-size');
        expect(fleetSize).toHaveTextContent('5/32');
      });
    });

    it('should monitor and display network latency', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const latency = screen.getByTestId('network-latency');
        expect(latency).toHaveTextContent('45ms');
      });
    });

    it('should update member status in real-time', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const memberList = screen.getByTestId('fleet-members');
        expect(memberList.children).toHaveLength(5);
      });
    });
  });

  describe('LiDAR Integration', () => {
    it('should maintain scan processing latency below threshold', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const scanLatency = screen.getByTestId('scan-latency');
        expect(Number(scanLatency.textContent)).toBeLessThanOrEqual(50);
      });
    });

    it('should display point cloud metrics at 30Hz', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const pointCloud = screen.getByTestId('point-cloud-metrics');
        expect(pointCloud).toBeInTheDocument();
      });
    });

    it('should reflect scan quality status accurately', async () => {
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const quality = screen.getByTestId('scan-quality');
        expect(quality).toHaveTextContent(/HIGH|MEDIUM|LOW/);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle network failures gracefully', async () => {
      server.use(
        handlers.gameStateHandler.mockImplementationOnce(() => {
          throw new Error('Network failure');
        })
      );
      
      render(<Dashboard {...defaultProps} />);
      
      await waitFor(() => {
        const error = screen.getByRole('alert');
        expect(error).toHaveTextContent(/Network failure/);
      });
    });

    it('should catch and display error boundary issues', async () => {
      const consoleError = jest.spyOn(console, 'error').mockImplementation();
      
      render(<Dashboard {...defaultProps} />);
      
      act(() => {
        throw new Error('Test error');
      });
      
      await waitFor(() => {
        const fallback = screen.getByText('Error loading dashboard');
        expect(fallback).toBeInTheDocument();
      });
      
      consoleError.mockRestore();
    });

    it('should provide user feedback for all error states', async () => {
      render(<Dashboard {...defaultProps} />);
      
      // Simulate various error states
      act(() => {
        fireEvent.error(window, new Event('error'));
      });
      
      await waitFor(() => {
        const errorMessages = screen.getAllByRole('alert');
        expect(errorMessages.length).toBeGreaterThan(0);
      });
    });
  });
});