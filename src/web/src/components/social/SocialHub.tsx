import React, { useEffect, useState, useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { Card } from '../common/Card';
import { FriendList, FriendListProps } from './FriendList';
import { NearbyPlayers, ConnectionQuality } from './NearbyPlayers';
import { WebRTCService } from '../../services/webrtc.service';
import { IUserProfile, UserStatusType, PrivacySettingsType } from '../../interfaces/user.interface';
import { FleetStatus, FleetNetworkStats } from '../../types/fleet.types';
import Automerge from 'automerge'; // @version 2.0.0
import { withHardwareSecurity } from '@tald/security'; // @version 1.0.0

// Styled components with HDR and power-aware optimizations
const SocialHubContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: calc(var(--spacing-unit) * 3);
  padding: calc(var(--spacing-unit) * 2);
  contain: content;
  will-change: transform;
  transform: var(--animation-gpu);

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const Section = styled(Card)`
  display: flex;
  flex-direction: column;
  gap: calc(var(--spacing-unit) * 2);
  height: min-content;
`;

const FleetStatus = styled.div<{ status: string }>`
  display: flex;
  align-items: center;
  gap: calc(var(--spacing-unit));
  padding: calc(var(--spacing-unit));
  background: ${({ status }) => 
    status === 'ACTIVE' ? 'color(display-p3 0 0.8 0 / 0.2)' :
    status === 'CONNECTING' ? 'color(display-p3 1 0.8 0 / 0.2)' :
    'color(display-p3 0.5 0.5 0.5 / 0.2)'};
  border-radius: calc(var(--spacing-unit) * 0.5);
`;

const NetworkStats = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: calc(var(--spacing-unit));
  padding: calc(var(--spacing-unit));
  background: color(display-p3 0.1 0.1 0.15);
  border-radius: calc(var(--spacing-unit) * 0.5);
`;

interface SocialHubProps {
  fleetId?: string;
  securityContext: string;
  privacySettings: PrivacySettingsType;
  region: string;
  powerMode?: 'performance' | 'balanced' | 'powersave';
  className?: string;
}

export const SocialHub: React.FC<SocialHubProps> = withHardwareSecurity(({
  fleetId,
  securityContext,
  privacySettings,
  region,
  powerMode = 'balanced',
  className
}) => {
  // State management
  const [friends, setFriends] = useState<IUserProfile[]>([]);
  const [networkStats, setNetworkStats] = useState<FleetNetworkStats>({
    averageLatency: 0,
    maxLatency: 0,
    minLatency: Number.MAX_VALUE,
    packetsLost: 0,
    bandwidth: 0,
    connectedPeers: 0,
    syncLatency: 0
  });
  const [fleetStatus, setFleetStatus] = useState<FleetStatus>(FleetStatus.INACTIVE);

  // CRDT state for fleet synchronization
  const [fleetDoc, setFleetDoc] = useState<Automerge.Doc<any>>(
    Automerge.init()
  );

  // WebRTC service initialization
  const webrtcService = useMemo(() => new WebRTCService(), []);

  // Network monitoring effect
  useEffect(() => {
    let monitoringInterval: NodeJS.Timer;

    const updateNetworkStats = async () => {
      try {
        const stats = await webrtcService.getNetworkStats();
        setNetworkStats(stats);
      } catch (error) {
        console.error('Network stats monitoring error:', error);
      }
    };

    // Adjust monitoring frequency based on power mode
    const interval = powerMode === 'powersave' ? 2000 : 1000;
    monitoringInterval = setInterval(updateNetworkStats, interval);

    return () => clearInterval(monitoringInterval);
  }, [webrtcService, powerMode]);

  // Fleet formation handler with security validation
  const handleFleetFormation = useCallback(async (player: IUserProfile, quality: ConnectionQuality) => {
    try {
      if (!privacySettings.gdprConsent) {
        throw new Error('GDPR consent required for fleet formation');
      }

      // Initialize fleet with CRDT
      const fleetState = Automerge.change(fleetDoc, doc => {
        doc.members = doc.members || [];
        doc.members.push({
          id: player.userId,
          joinedAt: Date.now()
        });
      });

      // Connect to fleet with WebRTC
      await webrtcService.connectToFleet(fleetId || crypto.randomUUID());
      await webrtcService.initializeCRDT(fleetState);

      setFleetDoc(fleetState);
      setFleetStatus(FleetStatus.ACTIVE);

    } catch (error) {
      console.error('Fleet formation error:', error);
      setFleetStatus(FleetStatus.INACTIVE);
    }
  }, [fleetDoc, fleetId, privacySettings.gdprConsent, webrtcService]);

  // Friend selection handler with privacy checks
  const handleFriendSelect = useCallback(async (friend: IUserProfile) => {
    if (!privacySettings.shareLocation) {
      console.warn('Location sharing disabled');
      return;
    }

    try {
      const stats = await webrtcService.getNetworkStats();
      if (stats.connectedPeers >= 32) {
        throw new Error('Maximum fleet size reached');
      }

      await handleFleetFormation(friend, {
        score: 1,
        status: 'excellent',
        latency: stats.averageLatency,
        bandwidth: stats.bandwidth
      });
    } catch (error) {
      console.error('Friend selection error:', error);
    }
  }, [handleFleetFormation, privacySettings.shareLocation, webrtcService]);

  return (
    <SocialHubContainer className={className}>
      <Section variant="elevated" powerSaveMode={powerMode === 'powersave'}>
        <h2>Nearby Players</h2>
        <FleetStatus status={fleetStatus}>
          Fleet Status: {fleetStatus}
        </FleetStatus>
        <NetworkStats>
          <div>Latency: {networkStats.averageLatency.toFixed(0)}ms</div>
          <div>Players: {networkStats.connectedPeers}/32</div>
          <div>Bandwidth: {(networkStats.bandwidth / 1000000).toFixed(1)}Mbps</div>
        </NetworkStats>
        <NearbyPlayers
          fleetId={fleetId}
          onPlayerSelect={handleFriendSelect}
          maxDistance={5}
          region={region}
          minQualityScore={0.7}
          gdprConsent={privacySettings.gdprConsent}
        />
      </Section>

      <Section variant="elevated" powerSaveMode={powerMode === 'powersave'}>
        <h2>Friends</h2>
        <FriendList
          friends={friends}
          onFriendSelect={handleFriendSelect}
          fleetId={fleetId}
          maxFleetSize={32}
          powerMode={powerMode}
        />
      </Section>
    </SocialHubContainer>
  );
});

export type { SocialHubProps };