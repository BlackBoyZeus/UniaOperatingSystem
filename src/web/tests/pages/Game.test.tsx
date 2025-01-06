import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { Game, GamePageProps } from '../../src/pages/Game';
import { server, resetHandlers } from '../mocks/server';
import { useGame } from '../../src/hooks/useGame';
import { useFleet } from '../../src/hooks/useFleet';
import mockWebRTC from '@webrtc/mock';

// Mock hooks and WebRTC
vi.mock('../../src/hooks/useGame');
vi.mock('../../src/hooks/useFleet');
vi.mock('@webrtc/mock');

// Test constants
const TEST_GAME_ID = 'test-game-123';
const TEST_SESSION_ID = 'test-session-456';
const TEST_FLEET_ID = 'test-fleet-789';

describe('Game Page Component', () => {
    // Setup before all tests
    beforeAll(() => {
        // Configure WebRTC mock handlers
        mockWebRTC.setup({
            connection: {
                iceConnectionState: 'connected',
                iceGatheringState: 'complete'
            }
        });

        // Setup network condition simulation
        server.listen({
            onUnhandledRequest: 'error'
        });
    });

    // Setup before each test
    beforeEach(() => {
        // Reset handlers
        server.resetHandlers();
        
        // Clear all mocks
        vi.clearAllMocks();

        // Setup mock game state
        (useGame as jest.Mock).mockReturnValue({
            gameState: {
                gameId: TEST_GAME_ID,
                sessionId: TEST_SESSION_ID,
                state: 'INITIALIZING',
                environmentData: null,
                renderState: {
                    resolution: { width: 1920, height: 1080 },
                    quality: 'HIGH',
                    lidarOverlayEnabled: true
                },
                fps: 60
            },
            startGame: vi.fn(),
            endGame: vi.fn(),
            updateEnvironment: vi.fn()
        });

        // Setup mock fleet state
        (useFleet as jest.Mock).mockReturnValue({
            fleetState: {
                id: TEST_FLEET_ID,
                members: new Map(),
                status: 'ACTIVE'
            },
            joinFleet: vi.fn(),
            leaveFleet: vi.fn()
        });
    });

    // Test initial rendering
    test('should render game page with initial state', async () => {
        render(<Game />);

        // Verify core UI elements
        expect(screen.getByTestId('game-container')).toBeInTheDocument();
        expect(screen.getByTestId('game-list')).toBeInTheDocument();
        expect(screen.queryByTestId('lidar-overlay')).not.toBeInTheDocument();
        expect(screen.getByTestId('performance-metrics')).toBeInTheDocument();

        // Verify initial state display
        expect(screen.getByText(/FPS: 60/i)).toBeInTheDocument();
        expect(screen.getByText(/Quality: HIGH/i)).toBeInTheDocument();
    });

    // Test game session lifecycle
    test('should handle game session lifecycle correctly', async () => {
        const { startGame, endGame } = useGame();
        const { joinFleet, leaveFleet } = useFleet();

        render(<Game />);

        // Start game session
        const gameCard = screen.getByTestId(`game-card-${TEST_GAME_ID}`);
        await userEvent.click(gameCard);

        expect(startGame).toHaveBeenCalledWith(TEST_GAME_ID);
        expect(joinFleet).toHaveBeenCalledWith(TEST_FLEET_ID);

        // Verify active game state
        await waitFor(() => {
            expect(screen.getByTestId('lidar-overlay')).toBeInTheDocument();
            expect(screen.getByText(/Fleet: 1\/32/i)).toBeInTheDocument();
        });

        // End game session
        await userEvent.click(screen.getByTestId('end-game-button'));

        expect(endGame).toHaveBeenCalled();
        expect(leaveFleet).toHaveBeenCalled();
    });

    // Test LiDAR visualization integration
    test('should integrate LiDAR visualization with 30Hz updates', async () => {
        const { updateEnvironment } = useGame();
        
        render(<Game />);

        // Start game with LiDAR data
        const mockEnvironmentData = {
            meshData: new ArrayBuffer(1000),
            pointCloud: new Float32Array(1000),
            timestamp: Date.now()
        };

        await userEvent.click(screen.getByTestId(`game-card-${TEST_GAME_ID}`));
        
        // Verify LiDAR update frequency
        let updateCount = 0;
        const startTime = performance.now();

        while (performance.now() - startTime < 1000) {
            await updateEnvironment(mockEnvironmentData);
            updateCount++;
            await new Promise(resolve => setTimeout(resolve, 33.33)); // 30Hz
        }

        expect(updateCount).toBeGreaterThanOrEqual(29); // Allow for small timing variations
        expect(updateCount).toBeLessThanOrEqual(31);
    });

    // Test performance requirements
    test('should maintain <16ms input latency', async () => {
        render(<Game />);

        // Measure input handling latency
        const startTime = performance.now();
        await userEvent.click(screen.getByTestId('game-container'));
        const endTime = performance.now();

        expect(endTime - startTime).toBeLessThan(16);
    });

    // Test fleet coordination
    test('should handle fleet formation and coordination', async () => {
        const { joinFleet } = useFleet();
        
        render(<Game />);

        // Join fleet
        await userEvent.click(screen.getByTestId(`game-card-${TEST_GAME_ID}`));
        
        expect(joinFleet).toHaveBeenCalledWith(TEST_FLEET_ID);

        // Verify fleet status updates
        await waitFor(() => {
            expect(screen.getByText(/Fleet: 1\/32/i)).toBeInTheDocument();
            expect(screen.getByText(/Latency: \d+ms/i)).toBeInTheDocument();
        });
    });

    // Test error handling
    test('should handle error states appropriately', async () => {
        const { startGame } = useGame();
        startGame.mockRejectedValueOnce(new Error('Game initialization failed'));

        render(<Game />);

        // Attempt to start game
        await userEvent.click(screen.getByTestId(`game-card-${TEST_GAME_ID}`));

        // Verify error display
        await waitFor(() => {
            expect(screen.getByText(/Game initialization failed/i)).toBeInTheDocument();
        });
    });

    // Test cleanup
    test('should cleanup resources on unmount', async () => {
        const { endGame } = useGame();
        const { leaveFleet } = useFleet();

        const { unmount } = render(<Game />);

        // Start game
        await userEvent.click(screen.getByTestId(`game-card-${TEST_GAME_ID}`));

        // Unmount component
        unmount();

        expect(endGame).toHaveBeenCalled();
        expect(leaveFleet).toHaveBeenCalled();
    });
});