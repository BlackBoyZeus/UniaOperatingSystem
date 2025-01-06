import React, { useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { IFleet, IFleetMember } from '../../interfaces/fleet.interface';
import { Card } from '../common/Card';
import { validateFleetSize, calculateFleetNetworkStats } from '../../utils/fleet.utils';
import { FleetStatus, MAX_LATENCY_THRESHOLD } from '../../types/fleet.types';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 */

interface FleetCardProps {
  fleet: IFleet;
  onJoin?: (fleetId: string) => Promise<void>;
  onLeave?: (fleetId: string) => Promise<void>;
  onManage?: (fleetId: string) => void;
  className?: string;
  interactive?: boolean;
  hdrEnabled?: boolean;
  powerMode?: 'performance' | 'balanced' | 'powersave';
  animationEnabled?: boolean;
}

const StyledFleetCard = styled.div<{
  networkQuality: number;
  powerMode: string;
  hdrEnabled: boolean;
}>`
  display: flex;
  flex-direction: column;
  gap: calc(var(--spacing-unit) * 1.5);
  min-width: 280px;
  max-width: 400px;
  transform: var(--animation-gpu);
  will-change: ${({ powerMode }) => 
    powerMode === 'powersave' ? 'auto' : 'transform, opacity'};
  contain: content;

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-family-gaming);
    font-feature-settings: var(--font-feature-gaming);
  }

  .stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: calc(var(--spacing-unit) * 1);
    font-family: var(--font-family-system);
  }

  .network-quality {
    display: flex;
    align-items: center;
    gap: calc(var(--spacing-unit) * 0.5);
    color: ${({ networkQuality, hdrEnabled }) => {
      if (networkQuality > 0.8) {
        return hdrEnabled 
          ? 'color(display-p3 0 1 0.5)'
          : 'rgb(0, 255, 128)';
      }
      if (networkQuality > 0.5) {
        return hdrEnabled
          ? 'color(display-p3 1 0.8 0)'
          : 'rgb(255, 204, 0)';
      }
      return hdrEnabled
        ? 'color(display-p3 1 0.3 0.3)'
        : 'rgb(255, 77, 77)';
    }};
    transition: color var(--animation-duration) ease;
  }

  .member-list {
    display: flex;
    flex-direction: column;
    gap: calc(var(--spacing-unit) * 0.5);
    max-height: 200px;
    overflow-y: auto;
    scrollbar-width: thin;
    contain: content;
  }

  .member-item {
    display: flex;
    align-items: center;
    gap: calc(var(--spacing-unit) * 1);
    padding: calc(var(--spacing-unit) * 0.5);
    border-radius: calc(var(--spacing-unit) * 0.5);
    background: color-mix(
      in display-p3,
      var(--color-surface) 80%,
      transparent
    );
  }

  .actions {
    display: flex;
    gap: calc(var(--spacing-unit) * 1);
    margin-top: calc(var(--spacing-unit) * 1);
  }
`;

export const FleetCard: React.FC<FleetCardProps> = ({
  fleet,
  onJoin,
  onLeave,
  onManage,
  className,
  interactive = true,
  hdrEnabled = true,
  powerMode = 'balanced',
  animationEnabled = true,
}) => {
  const networkStats = useMemo(() => 
    calculateFleetNetworkStats(fleet.members),
    [fleet.members]
  );

  const networkQuality = useMemo(() => {
    const latencyScore = 1 - (networkStats.averageLatency / MAX_LATENCY_THRESHOLD);
    const connectivityScore = networkStats.connectedPeers / fleet.maxDevices;
    return (latencyScore + connectivityScore) / 2;
  }, [networkStats, fleet.maxDevices]);

  const isValidFleet = useMemo(() => 
    validateFleetSize(fleet),
    [fleet]
  );

  const handleJoinFleet = useCallback(async (event: React.MouseEvent) => {
    event.preventDefault();
    if (!onJoin || !isValidFleet) return;

    const startTime = performance.now();
    try {
      await onJoin(fleet.id);
      const duration = performance.now() - startTime;
      console.debug(`Fleet join operation completed in ${duration}ms`);
    } catch (error) {
      console.error('Failed to join fleet:', error);
    }
  }, [fleet.id, onJoin, isValidFleet]);

  const handleLeaveFleet = useCallback(async (event: React.MouseEvent) => {
    event.preventDefault();
    if (!onLeave) return;

    try {
      await onLeave(fleet.id);
    } catch (error) {
      console.error('Failed to leave fleet:', error);
    }
  }, [fleet.id, onLeave]);

  const renderMemberList = useCallback((members: IFleetMember[]) => {
    return members.map(member => (
      <div key={member.id} className="member-item">
        <span>{member.id}</span>
        <span className="network-quality">
          {member.latency}ms
        </span>
      </div>
    ));
  }, []);

  return (
    <Card
      variant="elevated"
      interactive={interactive}
      hdrMode={hdrEnabled ? 'enabled' : 'disabled'}
      powerSaveMode={powerMode === 'powersave'}
      className={className}
    >
      <StyledFleetCard
        networkQuality={networkQuality}
        powerMode={powerMode}
        hdrEnabled={hdrEnabled}
      >
        <div className="header">
          <h3>{fleet.name}</h3>
          <span className="network-quality">
            {networkStats.averageLatency.toFixed(0)}ms
          </span>
        </div>

        <div className="stats">
          <div>Members: {fleet.members.length}/{fleet.maxDevices}</div>
          <div>Status: {fleet.status}</div>
        </div>

        <div className="member-list">
          {renderMemberList(fleet.members)}
        </div>

        {interactive && (
          <div className="actions">
            {fleet.status === FleetStatus.ACTIVE && onJoin && isValidFleet && (
              <button onClick={handleJoinFleet}>Join Fleet</button>
            )}
            {onLeave && (
              <button onClick={handleLeaveFleet}>Leave Fleet</button>
            )}
            {onManage && (
              <button onClick={() => onManage(fleet.id)}>Manage</button>
            )}
          </div>
        )}
      </StyledFleetCard>
    </Card>
  );
};

export type { FleetCardProps };