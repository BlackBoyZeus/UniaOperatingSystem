import { renderHook, act } from '@testing-library/react-hooks'; // @version ^8.0.1
import { waitFor } from '@testing-library/react'; // @version ^14.0.0
import '@testing-library/jest-dom'; // @version ^5.16.5
import * as Automerge from 'automerge'; // @version ^2.0.0

import { useFleet } from '../../src/hooks/useFleet';
import { server } from '../mocks/server';
import { 
    FleetStatus, 
    FleetRole, 
    FleetMessageType,
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD 
} from '../../src/types/fleet.types';
import type { 
    IFleet,
    IFleetMember,
    INetworkStats
} from '../../src/interfaces/fleet.interface';

describe('useFleet hook', () => {
    // Enhanced setup with WebRTC and network simulation
    beforeAll(() => {
        server.listen({
            onUnhandledRequest: 'error'
        });
    });

    afterAll(() => {
        server.close();
    });

    beforeEach(() => {
        server.resetHandlers();
        server.resetTelemetry();
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('should create a fleet with proper initialization', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            const fleet = await result.current.createFleet('Test Fleet', MAX_FLEET_SIZE);
            
            expect(fleet).toBeDefined();
            expect(fleet.id).toBeTruthy();
            expect(fleet.name).toBe('Test Fleet');
            expect(fleet.maxDevices).toBe(MAX_FLEET_SIZE);
            expect(fleet.status).toBe(FleetStatus.ACTIVE);
            expect(fleet.members).toHaveLength(0);
            expect(fleet.networkStats.averageLatency).toBeLessThanOrEqual(MAX_LATENCY_THRESHOLD);
        });

        expect(result.current.currentFleet).toBeTruthy();
        expect(result.current.error).toBeNull();
    });

    it('should handle CRDT state synchronization correctly', async () => {
        const { result } = renderHook(() => useFleet());
        
        await act(async () => {
            await result.current.createFleet('Sync Test Fleet', 3);
        });

        // Simulate multiple members joining
        const member1 = { id: 'member1', role: FleetRole.MEMBER };
        const member2 = { id: 'member2', role: FleetRole.MEMBER };

        await act(async () => {
            // Simulate concurrent state updates
            const state1 = Automerge.change(result.current.fleetState, 'Add member 1', doc => {
                doc.members = [member1];
            });

            const state2 = Automerge.change(result.current.fleetState, 'Add member 2', doc => {
                doc.members = [member2];
            });

            // Merge states
            const mergedState = Automerge.merge(state1, state2);
            
            // Verify state convergence
            expect(mergedState.members).toHaveLength(2);
            expect(mergedState.members).toContainEqual(member1);
            expect(mergedState.members).toContainEqual(member2);
        });
    });

    it('should manage leader election and failover', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            await result.current.createFleet('Leader Test Fleet', 3);
        });

        // Simulate leader setup
        const leader: IFleetMember = {
            id: 'leader1',
            role: FleetRole.LEADER,
            connection: {
                lastPing: Date.now(),
                connectionQuality: 1,
                retryCount: 0
            }
        };

        await act(async () => {
            // Add leader to fleet
            result.current.fleetMembers.set(leader.id, leader);

            // Simulate leader disconnection
            await result.current.handleLeaderFailover();

            // Verify new leader election
            const newLeader = Array.from(result.current.fleetMembers.values())
                .find(member => member.role === FleetRole.LEADER);

            expect(newLeader).toBeDefined();
            expect(newLeader?.id).not.toBe(leader.id);
        });
    });

    it('should monitor and optimize network performance', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            await result.current.createFleet('Network Test Fleet', MAX_FLEET_SIZE);
        });

        // Monitor initial network stats
        const initialStats = result.current.networkStats;

        // Simulate network degradation
        server.setNetworkProfile({
            latency: 100,
            jitter: 20,
            packetLoss: 0.05,
            bandwidth: 500000
        });

        await act(async () => {
            await result.current.optimizeConnections();
        });

        // Verify network optimization
        await waitFor(() => {
            const optimizedStats = result.current.networkStats;
            expect(optimizedStats.averageLatency).toBeLessThan(initialStats.averageLatency);
            expect(optimizedStats.packetsLost).toBeLessThan(initialStats.packetsLost);
        });
    });

    it('should handle fleet joining with proper state synchronization', async () => {
        const { result } = renderHook(() => useFleet());
        const fleetId = 'test-fleet-id';

        await act(async () => {
            const success = await result.current.joinFleet(fleetId);
            expect(success).toBe(true);
        });

        expect(result.current.currentFleet?.id).toBe(fleetId);
        expect(result.current.networkStats.connectedPeers).toBeGreaterThan(0);
        expect(result.current.error).toBeNull();
    });

    it('should handle graceful fleet departure', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            await result.current.createFleet('Departure Test Fleet', 3);
            await result.current.leaveFleet();
        });

        expect(result.current.currentFleet).toBeNull();
        expect(result.current.fleetMembers.size).toBe(0);
        expect(result.current.networkStats.connectedPeers).toBe(0);
    });

    it('should recover from network failures', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            await result.current.createFleet('Recovery Test Fleet', 3);
        });

        // Simulate network failure
        server.setNetworkProfile({
            latency: 1000,
            jitter: 50,
            packetLoss: 0.2,
            bandwidth: 100000
        });

        await act(async () => {
            // Attempt recovery
            await result.current.handleStateRecovery();
        });

        expect(result.current.currentFleet?.status).toBe(FleetStatus.ACTIVE);
        expect(result.current.error).toBeNull();
    });

    it('should enforce fleet size limits', async () => {
        const { result } = renderHook(() => useFleet());

        await act(async () => {
            await result.current.createFleet('Size Test Fleet', MAX_FLEET_SIZE);
        });

        // Attempt to exceed fleet size
        await act(async () => {
            for (let i = 0; i <= MAX_FLEET_SIZE + 1; i++) {
                const member: IFleetMember = {
                    id: `member-${i}`,
                    role: FleetRole.MEMBER,
                    connection: {
                        lastPing: Date.now(),
                        connectionQuality: 1,
                        retryCount: 0
                    }
                };
                result.current.fleetMembers.set(member.id, member);
            }
        });

        expect(result.current.fleetMembers.size).toBeLessThanOrEqual(MAX_FLEET_SIZE);
        expect(result.current.error).toBeTruthy();
    });
});