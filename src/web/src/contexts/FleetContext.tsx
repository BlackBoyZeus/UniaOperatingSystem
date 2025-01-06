// External imports - versions specified for security tracking
import React, { createContext, useContext, useEffect, useState } from 'react'; // @version 18.2.0
import * as Automerge from 'automerge'; // @version 2.0.0

// Internal imports
import { 
    IFleet, 
    IFleetMember, 
    IFleetConnection, 
    INetworkStats, 
    ILeaderState 
} from '../interfaces/fleet.interface';
import FleetService from '../services/fleet.service';
import { 
    FleetStatus, 
    FleetRole, 
    MAX_FLEET_SIZE, 
    DEFAULT_SYNC_INTERVAL 
} from '../types/fleet.types';

// Constants for fleet management
const NETWORK_MONITOR_INTERVAL = 1000;
const STATE_SYNC_INTERVAL = 50;
const LEADER_ELECTION_TIMEOUT = 5000;
const MIN_NETWORK_QUALITY = 0.8;
const MAX_SYNC_RETRY_COUNT = 3;

// Enhanced fleet context type definition
interface FleetContextType {
    currentFleet: IFleet | null;
    fleetMembers: Map<string, IFleetMember>;
    networkStats: INetworkStats;
    leaderState: ILeaderState;
    joinFleet: (fleetId: string) => Promise<boolean>;
    leaveFleet: () => Promise<void>;
    createFleet: (name: string, maxDevices: number) => Promise<IFleet>;
    initiateLeaderElection: () => Promise<void>;
    handleStateRecovery: () => Promise<void>;
}

// Create the context with enhanced type safety
const FleetContext = createContext<FleetContextType | null>(null);

// Custom hook for accessing fleet context with type safety
export const useFleetContext = (): FleetContextType => {
    const context = useContext(FleetContext);
    if (!context) {
        throw new Error('useFleetContext must be used within a FleetProvider');
    }
    return context;
};

// Enhanced fleet provider component
export const FleetProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
    const [fleetService] = useState(() => new FleetService());
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
    const [leaderState, setLeaderState] = useState<ILeaderState>({
        leaderId: null,
        backupLeaders: [],
        electionInProgress: false,
        lastElectionTime: null
    });
    const [peerConnections] = useState<Map<string, IFleetConnection>>(new Map());

    // Initialize CRDT document for state management
    const [fleetState, setFleetState] = useState(() => Automerge.init<IFleet>());

    useEffect(() => {
        const monitorInterval = setInterval(monitorNetworkQuality, NETWORK_MONITOR_INTERVAL);
        const syncInterval = setInterval(synchronizeFleetState, STATE_SYNC_INTERVAL);

        return () => {
            clearInterval(monitorInterval);
            clearInterval(syncInterval);
        };
    }, [currentFleet]);

    const createFleet = async (name: string, maxDevices: number = MAX_FLEET_SIZE): Promise<IFleet> => {
        try {
            const fleet = await fleetService.createFleet(name, maxDevices);
            setCurrentFleet(fleet);
            initializeFleetState(fleet);
            return fleet;
        } catch (error) {
            console.error('Failed to create fleet:', error);
            throw error;
        }
    };

    const joinFleet = async (fleetId: string): Promise<boolean> => {
        try {
            await fleetService.joinFleet(fleetId);
            const fleetState = await fleetService.syncFleetState();
            setCurrentFleet(fleetState);
            return true;
        } catch (error) {
            console.error('Failed to join fleet:', error);
            return false;
        }
    };

    const leaveFleet = async (): Promise<void> => {
        try {
            if (currentFleet) {
                await fleetService.leaveFleet();
                setCurrentFleet(null);
                setFleetMembers(new Map());
                setFleetState(Automerge.init<IFleet>());
            }
        } catch (error) {
            console.error('Failed to leave fleet:', error);
            throw error;
        }
    };

    const initiateLeaderElection = async (): Promise<void> => {
        try {
            setLeaderState(prev => ({ ...prev, electionInProgress: true }));
            await fleetService.initiateLeaderElection();
            await handleLeaderElection();
        } catch (error) {
            console.error('Leader election failed:', error);
            throw error;
        } finally {
            setLeaderState(prev => ({ ...prev, electionInProgress: false }));
        }
    };

    const handleStateRecovery = async (): Promise<void> => {
        try {
            const recoveredState = await fleetService.handleStateRecovery();
            setFleetState(recoveredState);
        } catch (error) {
            console.error('State recovery failed:', error);
            throw error;
        }
    };

    const monitorNetworkQuality = async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            const stats = await fleetService.monitorNetworkQuality();
            setNetworkStats(stats);

            if (stats.averageLatency > MAX_FLEET_SIZE || stats.packetsLost > 0.1) {
                await handleNetworkDegradation();
            }
        } catch (error) {
            console.error('Network monitoring failed:', error);
        }
    };

    const synchronizeFleetState = async (): Promise<void> => {
        if (!currentFleet) return;

        try {
            const newState = await fleetService.syncFleetState();
            setFleetState(prevState => Automerge.merge(prevState, newState));
        } catch (error) {
            console.error('State synchronization failed:', error);
        }
    };

    const value: FleetContextType = {
        currentFleet,
        fleetMembers,
        networkStats,
        leaderState,
        joinFleet,
        leaveFleet,
        createFleet,
        initiateLeaderElection,
        handleStateRecovery
    };

    return (
        <FleetContext.Provider value={value}>
            {children}
        </FleetContext.Provider>
    );
};

export default FleetContext;