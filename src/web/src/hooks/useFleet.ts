import { useState, useEffect, useCallback, useRef } from 'react'; // @version 18.2.0
import * as Automerge from 'automerge'; // @version 2.0.0

import { 
    IFleet, 
    IFleetMember, 
    IFleetConnection, 
    INetworkStats, 
    IRegionalRoute 
} from '../interfaces/fleet.interface';
import { FleetService } from '../services/fleet.service';
import { useFleetContext } from '../contexts/FleetContext';

// Constants for fleet management
const NETWORK_MONITOR_INTERVAL = 1000;
const STATE_SYNC_INTERVAL = 50;
const MAX_RETRY_ATTEMPTS = 3;
const LATENCY_THRESHOLD = 50;
const BANDWIDTH_THRESHOLD = 1000000;
const CONNECTION_POOL_SIZE = 64;

/**
 * Enhanced custom hook for managing fleet operations with CRDT-based state sync,
 * network quality monitoring, and regional routing optimization
 */
export const useFleet = () => {
    // Context and service initialization
    const fleetContext = useFleetContext();
    const [fleetService] = useState(() => new FleetService());
    
    // State management
    const [currentFleet, setCurrentFleet] = useState<IFleet | null>(null);
    const [fleetMembers, setFleetMembers] = useState<Map<string, IFleetMember>>(new Map());
    const [networkStats, setNetworkStats] = useState<INetworkStats>({
        averageLatency: 0,
        maxLatency: 0,
        minLatency: Number.MAX_VALUE,
        packetsLost: 0,
        bandwidth: 0,
        connectedPeers: 0,
        syncLatency: 0
    });
    const [regionalRoutes, setRegionalRoutes] = useState<Map<string, IRegionalRoute>>(new Map());
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<Error | null>(null);

    // CRDT state management
    const fleetState = useRef(Automerge.init<IFleet>());
    const retryAttempts = useRef<Map<string, number>>(new Map());
    const connectionPool = useRef<Map<string, IFleetConnection>>(new Map());

    // Initialize monitoring and sync intervals
    useEffect(() => {
        if (!currentFleet) return;

        const monitorInterval = setInterval(monitorNetworkQuality, NETWORK_MONITOR_INTERVAL);
        const syncInterval = setInterval(synchronizeFleetState, STATE_SYNC_INTERVAL);

        return () => {
            clearInterval(monitorInterval);
            clearInterval(syncInterval);
        };
    }, [currentFleet]);

    /**
     * Creates a new fleet with advanced monitoring and failover capabilities
     */
    const createFleet = useCallback(async (name: string, maxDevices: number): Promise<IFleet> => {
        setIsLoading(true);
        setError(null);

        try {
            const fleet = await fleetService.createFleet(name, maxDevices);
            setCurrentFleet(fleet);
            
            // Initialize CRDT state
            fleetState.current = Automerge.change(fleetState.current, 'Initialize fleet', doc => {
                Object.assign(doc, fleet);
            });

            await initializeFleetConnections(fleet);
            return fleet;

        } catch (error) {
            const err = error instanceof Error ? error : new Error('Failed to create fleet');
            setError(err);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [fleetService]);

    /**
     * Joins an existing fleet with state synchronization
     */
    const joinFleet = useCallback(async (fleetId: string): Promise<boolean> => {
        setIsLoading(true);
        setError(null);

        try {
            await fleetService.joinFleet(fleetId);
            const syncedState = await fleetService.syncFleetState();
            
            fleetState.current = Automerge.merge(fleetState.current, syncedState);
            setCurrentFleet(syncedState);
            
            await optimizeConnections();
            return true;

        } catch (error) {
            const err = error instanceof Error ? error : new Error('Failed to join fleet');
            setError(err);
            return false;
        } finally {
            setIsLoading(false);
        }
    }, [fleetService]);

    /**
     * Leaves current fleet with graceful connection termination
     */
    const leaveFleet = useCallback(async (): Promise<void> => {
        if (!currentFleet) return;

        setIsLoading(true);
        setError(null);

        try {
            await fleetService.leaveFleet();
            cleanupConnections();
            
            setCurrentFleet(null);
            setFleetMembers(new Map());
            fleetState.current = Automerge.init<IFleet>();

        } catch (error) {
            const err = error instanceof Error ? error : new Error('Failed to leave fleet');
            setError(err);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [currentFleet, fleetService]);

    /**
     * Optimizes fleet connections based on network quality and regional routing
     */
    const optimizeConnections = useCallback(async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            const routes = await fleetService.optimizeRegionalRoutes();
            setRegionalRoutes(new Map(routes));

            // Prune and optimize connection pool
            if (connectionPool.current.size > CONNECTION_POOL_SIZE) {
                const sortedConnections = Array.from(connectionPool.current.entries())
                    .sort(([, a], [, b]) => 
                        (a.networkStats?.latency || Infinity) - (b.networkStats?.latency || Infinity));
                
                const optimizedPool = new Map(sortedConnections.slice(0, CONNECTION_POOL_SIZE));
                connectionPool.current = optimizedPool;
            }

        } catch (error) {
            console.error('Connection optimization failed:', error);
        }
    }, [currentFleet, fleetService]);

    /**
     * Handles leader failover in case of disconnection or performance degradation
     */
    const handleLeaderFailover = useCallback(async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            await fleetService.handleLeaderElection();
            const newState = await fleetService.syncFleetState();
            
            fleetState.current = Automerge.merge(fleetState.current, newState);
            setCurrentFleet(newState);

        } catch (error) {
            console.error('Leader failover failed:', error);
            throw error;
        }
    }, [currentFleet, fleetService]);

    /**
     * Monitors network quality and triggers optimizations when needed
     */
    const monitorNetworkQuality = async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            const stats = await fleetService.monitorNetworkStats();
            setNetworkStats(stats);

            if (stats.averageLatency > LATENCY_THRESHOLD || 
                stats.bandwidth < BANDWIDTH_THRESHOLD) {
                await optimizeConnections();
            }

        } catch (error) {
            console.error('Network monitoring failed:', error);
        }
    };

    /**
     * Synchronizes fleet state using CRDT
     */
    const synchronizeFleetState = async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            const newState = await fleetService.syncFleetState();
            fleetState.current = Automerge.merge(fleetState.current, newState);
            setCurrentFleet(newState);

        } catch (error) {
            console.error('State synchronization failed:', error);
        }
    };

    /**
     * Initializes fleet connections with connection pooling
     */
    const initializeFleetConnections = async (fleet: IFleet): Promise<void> => {
        try {
            const connections = await fleetService.initializeFleetConnections(fleet);
            connectionPool.current = new Map(connections);
            setFleetMembers(new Map(fleet.members.map(member => [member.id, member])));

        } catch (error) {
            console.error('Connection initialization failed:', error);
            throw error;
        }
    };

    /**
     * Cleans up connections and resources
     */
    const cleanupConnections = (): void => {
        connectionPool.current.clear();
        retryAttempts.current.clear();
    };

    return {
        currentFleet,
        fleetMembers,
        networkStats,
        regionalRoutes,
        createFleet,
        joinFleet,
        leaveFleet,
        optimizeConnections,
        handleLeaderFailover,
        isLoading,
        error
    };
};

export default useFleet;