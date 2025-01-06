import { z } from 'zod'; // ^3.22.0
import { Automerge } from 'automerge'; // ^2.0.0

import { 
    IFleet, 
    IFleetMember,
    IFleetSchema,
    IFleetMemberSchema 
} from '../interfaces/fleet.interface';

import {
    FleetStatus,
    FleetRole,
    FleetNetworkStats,
    FleetMemberConnection,
    FleetCRDTState,
    MAX_FLEET_SIZE,
    MAX_LATENCY_THRESHOLD,
    MIN_LEADER_SCORE
} from '../types/fleet.types';

// Global constants for fleet management
const MIN_FLEET_SIZE = 2;
const NETWORK_STATS_INTERVAL = 1000; // ms
const MIN_LEADER_BACKUP_COUNT = 2;
const CRDT_SYNC_TIMEOUT = 5000; // ms
const QUALITY_CHECK_INTERVAL = 2000; // ms

/**
 * Validates fleet size and configuration including leader backup requirements
 * @param fleet The fleet to validate
 * @returns boolean indicating if fleet configuration is valid
 */
export function validateFleetSize(fleet: IFleet): boolean {
    try {
        // Validate fleet structure using zod schema
        IFleetSchema.parse(fleet);

        const memberCount = fleet.members.length;
        
        // Basic size validation
        if (memberCount < MIN_FLEET_SIZE || memberCount > MAX_FLEET_SIZE) {
            return false;
        }

        // Validate against fleet's maxDevices
        if (memberCount > fleet.maxDevices) {
            return false;
        }

        // Validate backup leader configuration
        const backupLeaders = fleet.members.filter(
            member => member.role === FleetRole.BACKUP_LEADER
        );

        if (backupLeaders.length < MIN_LEADER_BACKUP_COUNT) {
            return false;
        }

        // Validate leader quality metrics
        const leaderQualityValid = backupLeaders.every(leader => 
            leader.connectionQuality.stability >= MIN_LEADER_SCORE &&
            leader.latency <= MAX_LATENCY_THRESHOLD
        );

        return leaderQualityValid;
    } catch (error) {
        console.error('Fleet validation error:', error);
        return false;
    }
}

/**
 * Calculates comprehensive network statistics for the fleet
 * @param members Array of fleet members
 * @returns FleetNetworkStats object with detailed metrics
 */
export function calculateFleetNetworkStats(members: IFleetMember[]): FleetNetworkStats {
    if (!members.length) {
        return {
            averageLatency: 0,
            maxLatency: 0,
            minLatency: 0,
            packetsLost: 0,
            bandwidth: 0,
            connectedPeers: 0,
            syncLatency: 0
        };
    }

    // Calculate weighted latency metrics
    const latencies = members.map(m => m.latency);
    const weightedLatencies = latencies.map((latency, idx) => {
        const weight = members[idx].connectionQuality.stability;
        return latency * weight;
    });

    const stats: FleetNetworkStats = {
        averageLatency: weightedLatencies.reduce((a, b) => a + b, 0) / members.length,
        maxLatency: Math.max(...latencies),
        minLatency: Math.min(...latencies),
        packetsLost: members.reduce((total, m) => total + m.connection.retryCount, 0),
        bandwidth: members.reduce((total, m) => total + m.connectionQuality.signalStrength * 1000, 0),
        connectedPeers: members.filter(m => m.connection.connectionQuality > 0.5).length,
        syncLatency: calculateSyncLatency(members)
    };

    return stats;
}

/**
 * Validates fleet member configuration and connection quality
 * @param member Fleet member to validate
 * @returns boolean indicating if member meets quality requirements
 */
export function validateFleetMember(member: IFleetMember): boolean {
    try {
        // Validate member structure using zod schema
        IFleetMemberSchema.parse(member);

        // Validate connection quality
        const qualityValid = member.connectionQuality.stability >= 0.6 &&
                           member.connectionQuality.reliability >= 0.7 &&
                           member.connectionQuality.signalStrength >= 0.5;

        if (!qualityValid) {
            return false;
        }

        // Validate latency requirements
        if (member.latency > MAX_LATENCY_THRESHOLD) {
            return false;
        }

        // Additional role-specific validation
        if (member.role === FleetRole.LEADER || member.role === FleetRole.BACKUP_LEADER) {
            return validateLeaderRequirements(member);
        }

        // Validate CRDT operation timestamp
        const crdtTimeValid = Date.now() - member.lastCRDTOperation.timestamp <= CRDT_SYNC_TIMEOUT;

        return crdtTimeValid;
    } catch (error) {
        console.error('Member validation error:', error);
        return false;
    }
}

/**
 * Formats fleet state for CRDT synchronization with quality metrics
 * @param fleet Fleet to format
 * @returns FleetCRDTState object with formatted state
 */
export function formatFleetState(fleet: IFleet): FleetCRDTState {
    const doc = Automerge.init<FleetCRDTState>();
    
    return Automerge.change(doc, 'Format fleet state', doc => {
        doc.id = fleet.id;
        doc.name = fleet.name;
        doc.status = fleet.status;
        doc.networkQuality = calculateNetworkQuality(fleet);
        doc.members = fleet.members.map(member => ({
            id: member.id,
            role: member.role,
            quality: {
                connection: member.connectionQuality.stability,
                sync: member.lastCRDTOperation.timestamp,
                leader: member.role === FleetRole.LEADER || member.role === FleetRole.BACKUP_LEADER
                    ? calculateLeaderScore(member)
                    : 0
            }
        }));
        doc.timestamp = Date.now();
    });
}

/**
 * Helper function to calculate sync latency across fleet members
 */
function calculateSyncLatency(members: IFleetMember[]): number {
    const syncTimes = members.map(m => m.lastCRDTOperation.timestamp);
    const latest = Math.max(...syncTimes);
    const earliest = Math.min(...syncTimes);
    return latest - earliest;
}

/**
 * Helper function to validate leader-specific requirements
 */
function validateLeaderRequirements(member: IFleetMember): boolean {
    return member.connectionQuality.stability >= MIN_LEADER_SCORE &&
           member.connectionQuality.reliability >= MIN_LEADER_SCORE &&
           member.latency <= MAX_LATENCY_THRESHOLD * 0.8;
}

/**
 * Helper function to calculate overall network quality score
 */
function calculateNetworkQuality(fleet: IFleet): number {
    const latencyScore = 1 - (fleet.networkStats.averageLatency / MAX_LATENCY_THRESHOLD);
    const connectivityScore = fleet.networkStats.connectedPeers / fleet.maxDevices;
    const syncScore = 1 - (fleet.networkStats.syncLatency / CRDT_SYNC_TIMEOUT);
    
    return (latencyScore + connectivityScore + syncScore) / 3;
}

/**
 * Helper function to calculate leader eligibility score
 */
function calculateLeaderScore(member: IFleetMember): number {
    const latencyScore = 1 - (member.latency / MAX_LATENCY_THRESHOLD);
    const qualityScore = (
        member.connectionQuality.stability +
        member.connectionQuality.reliability +
        member.connectionQuality.signalStrength
    ) / 3;
    
    return (latencyScore + qualityScore) / 2;
}