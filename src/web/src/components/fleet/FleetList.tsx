import React, { useCallback, useEffect, useMemo, useState } from 'react'; // @version 18.2.0
import styled from '@emotion/styled'; // @version 11.11.0
import { VirtualList } from 'react-window'; // @version 1.8.9

import { FleetCard, FleetCardProps } from './FleetCard';
import { IFleet } from '../../interfaces/fleet.interface';
import { useFleet } from '../../hooks/useFleet';

const StyledFleetList = styled.div<{
  powerSaveMode: boolean;
  hdrEnabled: boolean;
}>`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: calc(var(--spacing-unit) * 2);
  padding: calc(var(--spacing-unit) * 2);
  width: 100%;
  max-width: var(--layout-max-width);
  margin: 0 auto;
  transform: var(--animation-gpu);
  will-change: ${({ powerSaveMode }) => 
    powerSaveMode ? 'auto' : 'transform, opacity'};
  contain: content;
  isolation: isolate;

  /* HDR-aware color adjustments */
  ${({ hdrEnabled }) => hdrEnabled && `
    @media (dynamic-range: high) {
      color-scheme: dark;
      color-space: display-p3;
      color-gamut: p3;
    }
  `}

  /* Power-save mode optimizations */
  ${({ powerSaveMode }) => powerSaveMode && `
    transition: all var(--animation-duration-power-save);
    animation: none;
    will-change: auto;
  `}
`;

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
  width: 100%;
  transform: var(--animation-gpu);
  backface-visibility: hidden;
`;

interface FleetListProps {
  fleets: IFleet[];
  onManageFleet: (fleetId: string) => void;
  className?: string;
  loading?: boolean;
  networkQuality?: number;
  powerSaveMode?: boolean;
  hdrEnabled?: boolean;
  region?: string;
}

export const FleetList: React.FC<FleetListProps> = ({
  fleets,
  onManageFleet,
  className,
  loading = false,
  networkQuality = 1,
  powerSaveMode = false,
  hdrEnabled = true,
  region = 'na'
}) => {
  const { joinFleet, leaveFleet, monitorNetworkQuality, handleStateRecovery } = useFleet();
  const [retryAttempts] = useState<Map<string, number>>(new Map());

  // Memoized fleet sorting based on network quality and region
  const sortedFleets = useMemo(() => {
    return [...fleets].sort((a, b) => {
      // Prioritize fleets in the same region
      if (a.region === region && b.region !== region) return -1;
      if (b.region === region && a.region !== region) return 1;

      // Then sort by network quality
      return (b.networkQuality?.connectionScore || 0) - (a.networkQuality?.connectionScore || 0);
    });
  }, [fleets, region]);

  // Handle fleet join with retry logic and error recovery
  const handleJoinFleet = useCallback(async (fleetId: string) => {
    try {
      const attempts = retryAttempts.get(fleetId) || 0;
      if (attempts >= 3) {
        console.error(`Max retry attempts reached for fleet ${fleetId}`);
        return;
      }

      const fleet = fleets.find(f => f.id === fleetId);
      if (!fleet) return;

      // Validate network quality before joining
      const currentQuality = await monitorNetworkQuality();
      if (currentQuality < 0.7) {
        console.warn('Network quality too low for fleet join');
        return;
      }

      const success = await joinFleet(fleetId);
      if (!success) {
        retryAttempts.set(fleetId, attempts + 1);
        throw new Error('Failed to join fleet');
      }

    } catch (error) {
      console.error('Fleet join error:', error);
      await handleStateRecovery();
    }
  }, [fleets, joinFleet, monitorNetworkQuality, handleStateRecovery, retryAttempts]);

  // Handle fleet leave with cleanup
  const handleLeaveFleet = useCallback(async (fleetId: string) => {
    try {
      await leaveFleet();
      retryAttempts.delete(fleetId);
    } catch (error) {
      console.error('Fleet leave error:', error);
    }
  }, [leaveFleet]);

  // Monitor network quality for active fleets
  useEffect(() => {
    const monitorInterval = setInterval(async () => {
      await monitorNetworkQuality();
    }, 1000);

    return () => clearInterval(monitorInterval);
  }, [monitorNetworkQuality]);

  if (loading) {
    return (
      <LoadingContainer>
        <div>Loading Fleets...</div>
      </LoadingContainer>
    );
  }

  return (
    <StyledFleetList
      className={className}
      powerSaveMode={powerSaveMode}
      hdrEnabled={hdrEnabled}
      data-fleet-count={fleets.length}
    >
      {sortedFleets.map(fleet => (
        <FleetCard
          key={fleet.id}
          fleet={fleet}
          onJoin={() => handleJoinFleet(fleet.id)}
          onLeave={() => handleLeaveFleet(fleet.id)}
          onManage={() => onManageFleet(fleet.id)}
          interactive={networkQuality >= 0.7}
          hdrEnabled={hdrEnabled}
          powerMode={powerSaveMode ? 'powersave' : 'balanced'}
          animationEnabled={!powerSaveMode}
        />
      ))}
    </StyledFleetList>
  );
};

export type { FleetListProps };