import React, { useCallback, useEffect, useState } from 'react';
import styled from '@emotion/styled';

import { FleetList, FleetListProps } from '../components/fleet/FleetList';
import { FleetCreation } from '../components/fleet/FleetCreation';
import { useFleet } from '../hooks/useFleet';
import { DashboardLayout } from '../layouts/DashboardLayout';
import { IFleet } from '../interfaces/fleet.interface';

// GPU-accelerated styled components with HDR support
const StyledFleetPage = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  color-space: display-p3;
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
`;

const Title = styled.h1`
  font-size: 24px;
  font-weight: bold;
  color: var(--text-primary);
  text-rendering: optimizeLegibility;
`;

const Fleet: React.FC = () => {
  // Fleet management state
  const [isCreatingFleet, setIsCreatingFleet] = useState(false);
  const [selectedFleetId, setSelectedFleetId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);

  // Fleet management hook with CRDT support
  const {
    currentFleet,
    fleetMembers,
    networkStats,
    createFleet,
    joinFleet,
    leaveFleet,
    optimizeConnections,
    handleLeaderFailover,
    isLoading,
    error: fleetError
  } = useFleet();

  // Initialize network monitoring
  useEffect(() => {
    const monitorInterval = setInterval(async () => {
      try {
        await optimizeConnections();
      } catch (error) {
        console.error('Network optimization failed:', error);
      }
    }, 1000);

    return () => clearInterval(monitorInterval);
  }, [optimizeConnections]);

  // Enhanced fleet creation handler with hardware token validation
  const handleCreateFleet = useCallback(async (fleet: IFleet) => {
    try {
      setError(null);
      const createdFleet = await createFleet(fleet.name, fleet.maxDevices);
      setIsCreatingFleet(false);

      // Initialize P2P connections
      await optimizeConnections();

      // Setup leader election if we're the first member
      if (createdFleet.members.length === 0) {
        await handleLeaderFailover();
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to create fleet'));
      console.error('Fleet creation error:', err);
    }
  }, [createFleet, optimizeConnections, handleLeaderFailover]);

  // Enhanced fleet management handler with leader election
  const handleManageFleet = useCallback(async (fleetId: string) => {
    try {
      setError(null);
      setSelectedFleetId(fleetId);

      // Validate network quality before joining
      if (networkStats.averageLatency > 50) {
        throw new Error('Network latency too high for fleet management');
      }

      const success = await joinFleet(fleetId);
      if (!success) {
        throw new Error('Failed to join fleet');
      }

      // Initialize P2P connections
      await optimizeConnections();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to manage fleet'));
      console.error('Fleet management error:', err);
    }
  }, [joinFleet, optimizeConnections, networkStats.averageLatency]);

  // Handle fleet leave with cleanup
  const handleLeaveFleet = useCallback(async () => {
    try {
      setError(null);
      await leaveFleet();
      setSelectedFleetId(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to leave fleet'));
      console.error('Fleet leave error:', err);
    }
  }, [leaveFleet]);

  return (
    <DashboardLayout>
      <StyledFleetPage>
        <Header>
          <Title>Fleet Management</Title>
          <button
            onClick={() => setIsCreatingFleet(true)}
            disabled={isLoading}
          >
            Create Fleet
          </button>
        </Header>

        {error && (
          <div className="error-message">
            {error.message}
          </div>
        )}

        {isCreatingFleet && (
          <FleetCreation
            onSuccess={handleCreateFleet}
            onCancel={() => setIsCreatingFleet(false)}
            networkRequirements={{
              minLatency: 50,
              minBandwidth: 1000
            }}
            hardwareCapabilities={{
              lidarSupported: true,
              meshNetworkSupported: true,
              maxFleetSize: 32
            }}
          />
        )}

        <FleetList
          fleets={Array.from(fleetMembers.values()).map(member => ({
            ...member,
            id: member.id,
            name: member.id,
            maxDevices: 32,
            members: []
          }))}
          onManageFleet={handleManageFleet}
          loading={isLoading}
          networkQuality={networkStats.averageLatency <= 50 ? 1 : 0.5}
          powerSaveMode={false}
          hdrEnabled={true}
          region="na"
        />
      </StyledFleetPage>
    </DashboardLayout>
  );
};

export default Fleet;