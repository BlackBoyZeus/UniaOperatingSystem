import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import {
    validateFleetSize,
    calculateFleetNetworkStats,
    validateFleetMember,
    formatFleetState,
    validateFleetLeader,
    validateCRDTSync
} from '../../src/utils/fleet.utils';

import {
    IFleet,
    IFleetMember,
    IFleetMetrics,
    ICRDTState
} from '../../src/interfaces/fleet.interface';

import {
    FleetStatus,
    FleetRole,
    FleetNetworkStats,
    FleetQualityMetrics
} from '../../src/types/fleet.types';

describe('Fleet Utility Tests', () => {
    let mockFleet: IFleet;
    let mockMember: IFleetMember;

    beforeEach(() => {
        jest.useFakeTimers();
        
        mockMember = {
            id: '123e4567-e89b-12d3-a456-426614174000',
            deviceId: 'device-123',
            role: FleetRole.MEMBER,
            connection: {
                lastPing: Date.now(),
                connectionQuality: 0.9,
                retryCount: 0
            },
            latency: 25,
            connectionQuality: {
                signalStrength: 0.9,
                stability: 0.85,
                reliability: 0.95
            },
            lastCRDTOperation: {
                timestamp: Date.now(),
                type: 'update',
                payload: {}
            }
        };

        mockFleet = {
            id: '123e4567-e89b-12d3-a456-426614174001',
            name: 'Test Fleet',
            maxDevices: 32,
            members: [{ ...mockMember }],
            status: FleetStatus.ACTIVE,
            networkStats: {
                averageLatency: 25,
                maxLatency: 45,
                minLatency: 15,
                packetsLost: 0,
                bandwidth: 1000,
                connectedPeers: 1,
                syncLatency: 10
            },
            qualityMetrics: {
                connectionScore: 0.9,
                syncSuccess: 100,
                leaderRedundancy: 1
            },
            backupLeaders: []
        };
    });

    describe('validateFleetSize', () => {
        it('should return true for fleet size within limits', () => {
            expect(validateFleetSize(mockFleet)).toBe(true);
        });

        it('should return false for fleet exceeding maximum size', () => {
            mockFleet.members = Array(33).fill(mockMember);
            expect(validateFleetSize(mockFleet)).toBe(false);
        });

        it('should return false for empty fleet', () => {
            mockFleet.members = [];
            expect(validateFleetSize(mockFleet)).toBe(false);
        });

        it('should respect custom maxDevices setting', () => {
            mockFleet.maxDevices = 16;
            mockFleet.members = Array(20).fill(mockMember);
            expect(validateFleetSize(mockFleet)).toBe(false);
        });

        it('should validate minimum leader backup requirements', () => {
            const backupLeader = { ...mockMember, role: FleetRole.BACKUP_LEADER };
            mockFleet.members = [mockMember, backupLeader];
            mockFleet.backupLeaders = [backupLeader.id];
            expect(validateFleetSize(mockFleet)).toBe(false); // Needs 2 backup leaders
        });

        it('should verify minimum active members for operation', () => {
            mockFleet.members = [{ ...mockMember, connection: { ...mockMember.connection, connectionQuality: 0.3 }}];
            expect(validateFleetSize(mockFleet)).toBe(false);
        });
    });

    describe('calculateFleetNetworkStats', () => {
        it('should calculate correct average latency', () => {
            const members = [
                { ...mockMember, latency: 20 },
                { ...mockMember, latency: 30 }
            ];
            const stats = calculateFleetNetworkStats(members);
            expect(stats.averageLatency).toBe(25);
        });

        it('should count connected members accurately', () => {
            const members = [
                mockMember,
                { ...mockMember, connection: { ...mockMember.connection, connectionQuality: 0.3 }}
            ];
            const stats = calculateFleetNetworkStats(members);
            expect(stats.connectedPeers).toBe(1);
        });

        it('should identify high latency connections', () => {
            const members = [
                mockMember,
                { ...mockMember, latency: 60 }
            ];
            const stats = calculateFleetNetworkStats(members);
            expect(stats.maxLatency).toBe(60);
        });

        it('should handle empty fleet gracefully', () => {
            const stats = calculateFleetNetworkStats([]);
            expect(stats).toEqual({
                averageLatency: 0,
                maxLatency: 0,
                minLatency: 0,
                packetsLost: 0,
                bandwidth: 0,
                connectedPeers: 0,
                syncLatency: 0
            });
        });

        it('should aggregate quality metrics correctly', () => {
            const members = [
                mockMember,
                { ...mockMember, connectionQuality: { ...mockMember.connectionQuality, stability: 0.7 }}
            ];
            const stats = calculateFleetNetworkStats(members);
            expect(stats.bandwidth).toBeGreaterThan(0);
        });
    });

    describe('validateFleetMember', () => {
        it('should validate connected member with good latency', () => {
            expect(validateFleetMember(mockMember)).toBe(true);
        });

        it('should reject member with excessive latency', () => {
            const highLatencyMember = { ...mockMember, latency: 60 };
            expect(validateFleetMember(highLatencyMember)).toBe(false);
        });

        it('should reject disconnected member', () => {
            const disconnectedMember = {
                ...mockMember,
                connection: { ...mockMember.connection, connectionQuality: 0.2 }
            };
            expect(validateFleetMember(disconnectedMember)).toBe(false);
        });

        it('should validate CRDT operations', () => {
            const outdatedCRDT = {
                ...mockMember,
                lastCRDTOperation: {
                    ...mockMember.lastCRDTOperation,
                    timestamp: Date.now() - 6000
                }
            };
            expect(validateFleetMember(outdatedCRDT)).toBe(false);
        });

        it('should verify leader eligibility', () => {
            const leaderMember = { ...mockMember, role: FleetRole.LEADER };
            expect(validateFleetMember(leaderMember)).toBe(true);
        });

        it('should check backup leader requirements', () => {
            const backupLeader = {
                ...mockMember,
                role: FleetRole.BACKUP_LEADER,
                connectionQuality: { ...mockMember.connectionQuality, stability: 0.5 }
            };
            expect(validateFleetMember(backupLeader)).toBe(false);
        });
    });

    describe('validateCRDTSync', () => {
        it('should verify state convergence', () => {
            const members = mockFleet.members.map(m => ({
                ...m,
                lastCRDTOperation: { ...m.lastCRDTOperation, timestamp: Date.now() }
            }));
            expect(validateCRDTSync(members)).toBe(true);
        });

        it('should handle network partitions', () => {
            const partitionedMembers = [
                mockMember,
                { ...mockMember, connection: { ...mockMember.connection, connectionQuality: 0.3 }}
            ];
            expect(validateCRDTSync(partitionedMembers)).toBe(false);
        });

        it('should manage concurrent operations', () => {
            const concurrentMembers = mockFleet.members.map(m => ({
                ...m,
                lastCRDTOperation: {
                    ...m.lastCRDTOperation,
                    timestamp: Date.now(),
                    type: 'concurrent_update'
                }
            }));
            expect(validateCRDTSync(concurrentMembers)).toBe(true);
        });

        it('should implement state recovery', () => {
            jest.advanceTimersByTime(6000);
            const recoveryMembers = mockFleet.members.map(m => ({
                ...m,
                lastCRDTOperation: {
                    ...m.lastCRDTOperation,
                    timestamp: Date.now() - 7000
                }
            }));
            expect(validateCRDTSync(recoveryMembers)).toBe(false);
        });
    });
});