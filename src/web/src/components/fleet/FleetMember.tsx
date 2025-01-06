import React, { useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { useThrottle } from 'react-use'; // @version 17.4.0

import { IFleetMember } from '../../interfaces/fleet.interface';
import { Card } from '../common/Card';
import { useFleet } from '../../hooks/useFleet';

// Enhanced props interface with power and HDR support
interface FleetMemberProps {
    member: IFleetMember;
    isCurrentDevice: boolean;
    onKick?: (memberId: string) => Promise<void>;
    powerMode: 'normal' | 'saving';
    hdrEnabled: boolean;
}

// GPU-accelerated styled components with HDR and power-aware optimizations
const StyledMemberCard = styled(Card)<{
    isCurrentDevice: boolean;
    powerMode: string;
    connectionQuality: number;
}>`
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: calc(var(--spacing-unit) * 1.5);
    margin: calc(var(--spacing-unit) * 1) 0;
    background: ${({ theme }) => theme.colors.surface};
    transform: translateZ(0);
    will-change: transform;
    backface-visibility: hidden;
    color-scheme: ${({ hdrEnabled }) => hdrEnabled ? 'display-p3' : 'srgb'};

    ${({ isCurrentDevice }) => isCurrentDevice && `
        border: 2px solid var(--color-primary);
        box-shadow: 0 0 8px var(--color-primary-hdr);
    `}

    ${({ powerMode }) => powerMode === 'saving' && `
        transition: none;
        will-change: auto;
        animation: none;
    `}

    ${({ connectionQuality }) => `
        opacity: ${0.5 + (connectionQuality * 0.5)};
    `}
`;

const StatusIndicator = styled.div<{ status: string; hdrEnabled: boolean }>`
    width: calc(var(--spacing-unit) * 1);
    height: calc(var(--spacing-unit) * 1);
    border-radius: 50%;
    margin-right: calc(var(--spacing-unit) * 1);
    background-color: ${({ status, hdrEnabled }) => 
        getConnectionStatusColor(status, hdrEnabled)};
    transition: background-color 150ms ease;
    will-change: background-color;
`;

const MemberInfo = styled.div`
    display: flex;
    align-items: center;
    flex: 1;
    font-family: var(--font-family-gaming);
    font-feature-settings: var(--font-feature-gaming);
`;

const LatencyIndicator = styled.div<{ latency: number; hdrEnabled: boolean }>`
    font-size: 12px;
    color: ${({ latency, hdrEnabled }) => 
        getLatencyColor(latency, hdrEnabled)};
    margin-left: auto;
    padding: 0 calc(var(--spacing-unit) * 1);
    display: flex;
    align-items: center;
    gap: calc(var(--spacing-unit) * 0.5);
`;

const KickButton = styled.button`
    background: none;
    border: none;
    color: var(--color-error);
    cursor: pointer;
    padding: calc(var(--spacing-unit) * 0.5);
    opacity: 0.8;
    transition: opacity 150ms ease;

    &:hover {
        opacity: 1;
    }
`;

// Utility function for HDR-aware connection status colors
const getConnectionStatusColor = (status: string, hdrEnabled: boolean): string => {
    const colors = {
        CONNECTED: hdrEnabled 
            ? 'color(display-p3 0 1 0.5)'
            : 'rgb(0, 255, 128)',
        CONNECTING: hdrEnabled
            ? 'color(display-p3 1 0.8 0)'
            : 'rgb(255, 204, 0)',
        DISCONNECTED: hdrEnabled
            ? 'color(display-p3 1 0 0)'
            : 'rgb(255, 0, 0)'
    };
    return colors[status] || colors.DISCONNECTED;
};

// Utility function for HDR-aware latency colors
const getLatencyColor = (latency: number, hdrEnabled: boolean): string => {
    if (latency <= 30) {
        return hdrEnabled 
            ? 'color(display-p3 0 1 0.5)'
            : 'rgb(0, 255, 128)';
    } else if (latency <= 50) {
        return hdrEnabled
            ? 'color(display-p3 1 0.8 0)'
            : 'rgb(255, 204, 0)';
    }
    return hdrEnabled
        ? 'color(display-p3 1 0 0)'
        : 'rgb(255, 0, 0)';
};

export const FleetMember: React.FC<FleetMemberProps> = ({
    member,
    isCurrentDevice,
    onKick,
    powerMode,
    hdrEnabled
}) => {
    const { currentFleet } = useFleet();
    
    // Throttle kick handler for performance
    const handleKick = useThrottle(
        useCallback(async (e: React.MouseEvent) => {
            e.preventDefault();
            if (!onKick) return;

            try {
                await onKick(member.id);
            } catch (error) {
                console.error('Failed to kick member:', error);
            }
        }, [member.id, onKick]),
        100
    );

    // Memoize connection quality calculation
    const connectionQuality = useMemo(() => {
        const { latency, connectionQuality: quality } = member;
        return Math.min(1, Math.max(0, 1 - (latency / 100))) * quality;
    }, [member.latency, member.connectionQuality]);

    return (
        <StyledMemberCard
            isCurrentDevice={isCurrentDevice}
            powerMode={powerMode}
            connectionQuality={connectionQuality}
            hdrEnabled={hdrEnabled}
            data-testid={`fleet-member-${member.id}`}
            aria-label={`Fleet member ${member.deviceId}`}
        >
            <MemberInfo>
                <StatusIndicator 
                    status={member.connection.status}
                    hdrEnabled={hdrEnabled}
                />
                <span>{member.deviceId}</span>
                {member.role === 'LEADER' && (
                    <span role="img" aria-label="Fleet Leader" style={{ marginLeft: 8 }}>
                        👑
                    </span>
                )}
            </MemberInfo>

            <LatencyIndicator 
                latency={member.latency}
                hdrEnabled={hdrEnabled}
            >
                {member.latency}ms
                {member.connectionQuality.packetLoss > 0 && (
                    <span role="img" aria-label="Packet Loss Warning">
                        ⚠️
                    </span>
                )}
            </LatencyIndicator>

            {currentFleet?.members.length > 1 && 
             !isCurrentDevice && 
             onKick && (
                <KickButton
                    onClick={handleKick}
                    aria-label="Kick member"
                    data-testid={`kick-member-${member.id}`}
                >
                    ✕
                </KickButton>
            )}
        </StyledMemberCard>
    );
};

export type { FleetMemberProps };