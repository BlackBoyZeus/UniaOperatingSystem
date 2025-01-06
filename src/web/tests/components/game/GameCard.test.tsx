import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mockPerformanceMetrics } from '@testing-library/react-hooks';

// Internal imports
import { GameCard, GameCardProps } from '../../../src/components/game/GameCard';
import { IWebGameState } from '../../../src/interfaces/game.interface';
import { handlers } from '../../mocks/handlers';

// Mock performance.now() for consistent timing tests
vi.spyOn(performance, 'now').mockImplementation(() => Date.now());

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
}));

// Mock initial game state
const mockGameState: IWebGameState = {
    gameId: 'test-game-123',
    sessionId: 'test-session-456',
    state: 'RUNNING',
    environmentData: {
        meshData: new ArrayBuffer(1000),
        pointCloud: new Float32Array(1200000),
        classifiedObjects: [],
        timestamp: Date.now()
    },
    renderState: {
        resolution: { width: 1920, height: 1080 },
        quality: 'HIGH',
        lidarOverlayEnabled: true
    },
    fps: 60
};

// Mock fleet state
const mockFleetState = {
    members: Array(3).fill(null).map((_, i) => ({
        id: `member-${i}`,
        name: `Player ${i}`,
        latency: 45
    })),
    maxSize: 32
};

describe('GameCard Component', () => {
    let props: GameCardProps;
    
    beforeEach(() => {
        props = {
            gameState: mockGameState,
            onJoinGame: vi.fn().mockResolvedValue(undefined),
            onLeaveGame: vi.fn().mockResolvedValue(undefined),
            isActive: false,
            fleetState: mockFleetState,
            environmentState: mockGameState.environmentData,
            isHDREnabled: true,
            powerMode: 0
        };

        // Reset performance metrics
        mockPerformanceMetrics.mockClear();
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('renders with initial state correctly', () => {
        render(<GameCard {...props} />);
        
        expect(screen.getByText(mockGameState.gameId)).toBeInTheDocument();
        expect(screen.getByText(`Session: ${mockGameState.sessionId}`)).toBeInTheDocument();
        expect(screen.getByText('3/32 Players')).toBeInTheDocument();
    });

    it('handles real-time state updates within performance constraints', async () => {
        const { rerender } = render(<GameCard {...props} />);
        
        const startTime = performance.now();
        
        // Simulate state update
        const updatedState = {
            ...mockGameState,
            environmentData: {
                ...mockGameState.environmentData,
                timestamp: Date.now()
            }
        };
        
        rerender(<GameCard {...props} gameState={updatedState} />);
        
        const updateTime = performance.now() - startTime;
        expect(updateTime).toBeLessThanOrEqual(16.67); // 60 FPS threshold
    });

    it('maintains performance during LiDAR visualization updates', async () => {
        const { rerender } = render(<GameCard {...props} />);
        
        // Monitor frame timing during updates
        const frameTimings: number[] = [];
        const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            frameTimings.push(entries[0].duration);
        });
        observer.observe({ entryTypes: ['frame'] });

        // Simulate multiple LiDAR updates
        for (let i = 0; i < 10; i++) {
            const updatedEnvironment = {
                ...mockGameState.environmentData,
                pointCloud: new Float32Array(1200000),
                timestamp: Date.now()
            };
            
            rerender(
                <GameCard 
                    {...props} 
                    environmentState={updatedEnvironment}
                />
            );
            
            await new Promise(resolve => requestAnimationFrame(resolve));
        }

        observer.disconnect();

        // Verify frame timing consistency
        const avgFrameTime = frameTimings.reduce((a, b) => a + b, 0) / frameTimings.length;
        expect(avgFrameTime).toBeLessThanOrEqual(16.67);
    });

    it('handles network latency simulation correctly', async () => {
        const { rerender } = render(<GameCard {...props} />);
        
        // Simulate network delay
        const joinGameWithLatency = vi.fn().mockImplementation(() => 
            new Promise(resolve => setTimeout(resolve, 45)) // 45ms latency
        );

        rerender(
            <GameCard 
                {...props}
                onJoinGame={joinGameWithLatency}
            />
        );

        fireEvent.click(screen.getByRole('button'));
        
        const startTime = performance.now();
        await waitFor(() => expect(joinGameWithLatency).toHaveBeenCalled());
        const responseTime = performance.now() - startTime;
        
        expect(responseTime).toBeLessThanOrEqual(50); // Max latency threshold
    });

    it('adapts rendering quality based on performance metrics', async () => {
        const { rerender } = render(<GameCard {...props} />);
        
        // Simulate performance degradation
        mockPerformanceMetrics({
            fps: 45,
            frameTime: 22,
            memoryUsage: 3800
        });

        rerender(<GameCard {...props} />);
        
        await waitFor(() => {
            expect(props.gameState.renderState.quality).toBe('MEDIUM');
        });
    });

    it('handles fleet state synchronization correctly', async () => {
        render(<GameCard {...props} isActive={true} />);
        
        // Verify fleet member count
        const fleetStatus = screen.getByText('3/32 Players');
        expect(fleetStatus).toBeInTheDocument();
        
        // Verify member list updates
        const memberList = screen.getByRole('list');
        expect(within(memberList).getAllByRole('listitem')).toHaveLength(3);
    });

    it('implements accessibility requirements', () => {
        render(<GameCard {...props} />);
        
        // Verify ARIA attributes
        expect(screen.getByRole('button')).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite');
    });

    it('handles error states appropriately', async () => {
        const onJoinGame = vi.fn().mockRejectedValue(new Error('Join failed'));
        
        render(
            <GameCard 
                {...props}
                onJoinGame={onJoinGame}
            />
        );

        fireEvent.click(screen.getByRole('button'));
        
        await waitFor(() => {
            expect(screen.getByRole('alert')).toBeInTheDocument();
            expect(screen.getByText(/Join failed/i)).toBeInTheDocument();
        });
    });

    it('optimizes memory usage during extended sessions', async () => {
        const { rerender } = render(<GameCard {...props} isActive={true} />);
        
        // Monitor memory usage
        const memoryUsage: number[] = [];
        const getMemoryUsage = () => (performance as any).memory?.usedJSHeapSize || 0;
        
        // Simulate extended session with multiple updates
        for (let i = 0; i < 10; i++) {
            const updatedState = {
                ...mockGameState,
                environmentData: {
                    ...mockGameState.environmentData,
                    pointCloud: new Float32Array(1200000),
                    timestamp: Date.now()
                }
            };
            
            rerender(<GameCard {...props} gameState={updatedState} />);
            memoryUsage.push(getMemoryUsage());
            
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Verify memory usage remains within bounds
        const maxMemory = Math.max(...memoryUsage);
        expect(maxMemory).toBeLessThanOrEqual(3800 * 1024 * 1024); // 3.8GB limit
    });
});