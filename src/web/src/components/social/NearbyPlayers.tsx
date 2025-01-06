import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { Card } from '../common/Card';
import useWebRTC from '../../hooks/useWebRTC';
import type { IUser } from '../../interfaces/user.interface';

// Version comments for external dependencies
/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 */

// Enhanced styled components with HDR and power-aware optimizations
const NearbyPlayersContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: calc(var(--spacing-unit) * 2);
  contain: content;
  will-change: transform;
  transform: var(--animation-gpu);
`;

const PlayerCard = styled(Card)<{ qualityScore: number }>`
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: calc(var(--spacing-unit) * 2);
  padding: calc(var(--spacing-unit) * 2);
  background: ${({ qualityScore }) => `
    color-mix(
      in display-p3,
      var(--color-surface),
      var(--color-primary) ${qualityScore * 10}%
    )
  `};
`;

const QualityIndicator = styled.div<{ quality: number }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: ${({ quality }) => {
    if (quality >= 0.9) return 'color(display-p3 0 1 0)';
    if (quality >= 0.7) return 'color(display-p3 1 1 0)';
    if (quality >= 0.5) return 'color(display-p3 1 0.5 0)';
    return 'color(display-p3 1 0 0)';
  }};
  transition: background var(--animation-duration);
`;

interface NearbyPlayersProps {
  fleetId?: string;
  onPlayerSelect: (player: IUser, quality: ConnectionQuality) => void;
  maxDistance: number;
  className?: string;
  region: string;
  minQualityScore: number;
  gdprConsent: boolean;
}

interface NearbyPlayer {
  user: IUser;
  distance: number;
  latency: number;
  qualityScore: number;
  deviceCapabilities: DeviceCapabilities;
  region: string;
}

interface ConnectionQuality {
  score: number;
  status: 'excellent' | 'good' | 'fair' | 'poor';
  latency: number;
  bandwidth: number;
}

const NearbyPlayers: React.FC<NearbyPlayersProps> = ({
  fleetId,
  onPlayerSelect,
  maxDistance,
  className,
  region,
  minQualityScore,
  gdprConsent
}) => {
  const [nearbyPlayers, setNearbyPlayers] = useState<NearbyPlayer[]>([]);
  const { connectToFleet, networkStats, connectionQuality } = useWebRTC();

  // Memoized filtered and sorted players
  const filteredPlayers = useMemo(() => {
    return nearbyPlayers
      .filter(player => (
        player.distance <= maxDistance &&
        player.qualityScore >= minQualityScore &&
        player.region === region &&
        (!fleetId || player.user.fleetId !== fleetId)
      ))
      .sort((a, b) => (
        b.qualityScore - a.qualityScore || 
        a.distance - b.distance
      ));
  }, [nearbyPlayers, maxDistance, minQualityScore, region, fleetId]);

  // Handle player discovery with quality monitoring
  const handlePlayerDiscovery = useCallback(async (discoveredPlayer: IUser) => {
    if (!gdprConsent) return;

    try {
      const quality = await measureConnectionQuality(discoveredPlayer);
      const distance = calculateDistance(discoveredPlayer);

      setNearbyPlayers(current => {
        const updated = current.filter(p => p.user.id !== discoveredPlayer.id);
        return [...updated, {
          user: discoveredPlayer,
          distance,
          latency: quality.latency,
          qualityScore: quality.score,
          deviceCapabilities: discoveredPlayer.deviceCapabilities,
          region
        }];
      });
    } catch (error) {
      console.error('Player discovery error:', error);
    }
  }, [gdprConsent, region]);

  // Handle player selection with quality validation
  const handlePlayerSelect = useCallback(async (player: NearbyPlayer) => {
    if (!gdprConsent || player.qualityScore < minQualityScore) return;

    try {
      // Validate device capabilities compatibility
      if (!validateDeviceCompatibility(player.deviceCapabilities)) {
        throw new Error('Device capabilities incompatible');
      }

      // Verify mesh network capacity
      if (networkStats.connectedPeers >= player.deviceCapabilities.maxFleetSize) {
        throw new Error('Fleet capacity reached');
      }

      const quality: ConnectionQuality = {
        score: player.qualityScore,
        status: getQualityStatus(player.qualityScore),
        latency: player.latency,
        bandwidth: networkStats.bandwidth
      };

      onPlayerSelect(player.user, quality);
    } catch (error) {
      console.error('Player selection error:', error);
    }
  }, [gdprConsent, minQualityScore, networkStats, onPlayerSelect]);

  // Setup discovery monitoring
  useEffect(() => {
    if (!gdprConsent) return;

    const discoveryInterval = setInterval(() => {
      monitorNearbyPlayers();
    }, 1000);

    return () => clearInterval(discoveryInterval);
  }, [gdprConsent]);

  // Cleanup stale players
  useEffect(() => {
    const cleanupInterval = setInterval(() => {
      setNearbyPlayers(current => 
        current.filter(player => 
          Date.now() - player.user.lastActive.getTime() < 5000
        )
      );
    }, 2500);

    return () => clearInterval(cleanupInterval);
  }, []);

  return (
    <NearbyPlayersContainer className={className}>
      {filteredPlayers.map(player => (
        <PlayerCard
          key={player.user.id}
          qualityScore={player.qualityScore}
          onClick={() => handlePlayerSelect(player)}
          interactive
          variant="elevated"
          data-testid={`player-card-${player.user.id}`}
        >
          <QualityIndicator quality={player.qualityScore} />
          <div>
            <h3>{player.user.username}</h3>
            <p>{Math.round(player.distance)}m away</p>
          </div>
          <div>
            <p>{Math.round(player.latency)}ms</p>
            <p>{getQualityStatus(player.qualityScore)}</p>
          </div>
        </PlayerCard>
      ))}
    </NearbyPlayersContainer>
  );
};

// Helper functions
const measureConnectionQuality = async (player: IUser): Promise<{ score: number; latency: number }> => {
  // Implementation would use WebRTC service to measure actual connection quality
  return { score: 0.9, latency: 45 };
};

const calculateDistance = (player: IUser): number => {
  // Implementation would use actual device location data
  return Math.random() * 10;
};

const validateDeviceCompatibility = (capabilities: DeviceCapabilities): boolean => {
  return capabilities.meshNetworkSupported && capabilities.lidarSupported;
};

const getQualityStatus = (score: number): 'excellent' | 'good' | 'fair' | 'poor' => {
  if (score >= 0.9) return 'excellent';
  if (score >= 0.7) return 'good';
  if (score >= 0.5) return 'fair';
  return 'poor';
};

const monitorNearbyPlayers = () => {
  // Implementation would use actual device discovery mechanism
};

export type { NearbyPlayersProps, ConnectionQuality };
export default NearbyPlayers;