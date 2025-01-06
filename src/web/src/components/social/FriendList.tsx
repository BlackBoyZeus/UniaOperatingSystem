import React, { useEffect, useState, useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { VirtualList } from 'react-window'; // @version 1.8.9

import { Card } from '../common/Card';
import { WebRTCService } from '../../services/webrtc.service';
import AuthService from '../../services/auth.service';
import { IUserProfile, UserStatusType } from '../../interfaces/user.interface';
import { FleetNetworkStats, FleetStatus } from '../../types/fleet.types';

// Props interfaces
interface FriendListProps {
  friends: IUserProfile[];
  onFriendSelect: (friend: IUserProfile) => void;
  className?: string;
  fleetId?: string;
  maxFleetSize?: number;
  powerMode: 'performance' | 'balanced' | 'powersave';
}

interface FriendItemProps {
  friend: IUserProfile;
  onSelect: (friend: IUserProfile) => void;
  isOnline: boolean;
  fleetStatus: FleetStatus;
  networkQuality: FleetNetworkStats;
  powerMode: FriendListProps['powerMode'];
}

// Styled components with HDR and power-aware optimizations
const StyledFriendList = styled(Card)`
  width: 100%;
  max-height: 600px;
  overflow: hidden;
  contain: content;
  will-change: transform;
  transform: var(--animation-gpu);

  @media (dynamic-range: high) {
    background: color(display-p3 0.15 0.15 0.2);
  }
`;

const FriendItem = styled.div<{ isOnline: boolean; powerMode: string }>`
  display: flex;
  align-items: center;
  padding: calc(var(--spacing-unit) * 1.5);
  border-bottom: 1px solid color(display-p3 1 1 1 / 0.1);
  cursor: pointer;
  transition: all ${({ powerMode }) => 
    powerMode === 'powersave' ? 'var(--animation-duration-power-save)' : 'var(--animation-duration)'};

  &:hover {
    background: color(display-p3 0.2 0.2 0.25);
    transform: ${({ powerMode }) => 
      powerMode === 'performance' ? 'var(--animation-gpu) scale(1.02)' : 'none'};
  }

  ${({ isOnline }) => isOnline && `
    border-left: 4px solid var(--color-primary);
    @media (dynamic-range: high) {
      border-left-color: var(--color-primary-hdr);
    }
  `}
`;

const StatusIndicator = styled.div<{ status: UserStatusType }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: var(--spacing-unit);
  background: ${({ status }) => {
    switch (status) {
      case UserStatusType.ONLINE:
        return 'color(display-p3 0 1 0)';
      case UserStatusType.IN_GAME:
        return 'color(display-p3 1 0.5 0)';
      case UserStatusType.IN_FLEET:
        return 'color(display-p3 0 0.8 1)';
      default:
        return 'color(display-p3 0.5 0.5 0.5)';
    }
  }};
`;

const FriendInfo = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: calc(var(--spacing-unit) * 0.5);
`;

const FriendName = styled.span`
  font-family: var(--font-family-gaming);
  color: var(--color-primary);
  font-weight: var(--font-weight-medium);
`;

const NetworkStats = styled.div<{ quality: number }>`
  font-size: 0.8em;
  color: ${({ quality }) => 
    quality > 0.8 ? 'color(display-p3 0 1 0)' : 
    quality > 0.5 ? 'color(display-p3 1 0.5 0)' : 
    'color(display-p3 1 0 0)'};
`;

export const FriendList: React.FC<FriendListProps> = ({
  friends,
  onFriendSelect,
  className,
  fleetId,
  maxFleetSize = 32,
  powerMode = 'balanced'
}) => {
  const [networkStats, setNetworkStats] = useState<FleetNetworkStats>({
    averageLatency: 0,
    maxLatency: 0,
    minLatency: 0,
    packetsLost: 0,
    bandwidth: 0,
    connectedPeers: 0,
    syncLatency: 0
  });

  const handleFriendSelect = useCallback(async (friend: IUserProfile) => {
    try {
      // Validate hardware token and fleet capacity
      const authState = await AuthService.validateHardwareToken(friend.userId);
      if (!authState) {
        throw new Error('Hardware token validation failed');
      }

      // Check fleet capacity
      const currentStats = await WebRTCService.getFleetStatus();
      if (currentStats.connectedPeers >= maxFleetSize) {
        throw new Error('Fleet capacity reached');
      }

      // Monitor network quality
      const latency = await WebRTCService.monitorLatency(friend.userId);
      if (latency > 50) {
        throw new Error('Network latency too high');
      }

      onFriendSelect(friend);
    } catch (error) {
      console.error('Friend selection error:', error);
    }
  }, [onFriendSelect, maxFleetSize]);

  // Network monitoring effect
  useEffect(() => {
    let intervalId: NodeJS.Timer;

    const updateNetworkStats = async () => {
      const stats = await WebRTCService.getNetworkStats();
      setNetworkStats(stats);
    };

    // Adjust monitoring frequency based on power mode
    const interval = powerMode === 'powersave' ? 2000 : 1000;
    intervalId = setInterval(updateNetworkStats, interval);

    return () => clearInterval(intervalId);
  }, [powerMode]);

  // Memoized virtual list for performance
  const VirtualizedList = useMemo(() => {
    const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const friend = friends[index];
      const isOnline = friend.status !== UserStatusType.OFFLINE;

      return (
        <FriendItem
          style={style}
          isOnline={isOnline}
          powerMode={powerMode}
          onClick={() => handleFriendSelect(friend)}
        >
          <StatusIndicator status={friend.status} />
          <FriendInfo>
            <FriendName>{friend.displayName}</FriendName>
            {isOnline && (
              <NetworkStats quality={networkStats.connectedPeers / maxFleetSize}>
                {networkStats.averageLatency.toFixed(0)}ms | {networkStats.bandwidth / 1000}kb/s
              </NetworkStats>
            )}
          </FriendInfo>
        </FriendItem>
      );
    };

    return (
      <VirtualList
        height={550}
        width="100%"
        itemCount={friends.length}
        itemSize={70}
        overscanCount={5}
      >
        {Row}
      </VirtualList>
    );
  }, [friends, networkStats, powerMode, handleFriendSelect, maxFleetSize]);

  return (
    <StyledFriendList
      className={className}
      variant="elevated"
      powerSaveMode={powerMode === 'powersave'}
      hdrMode="auto"
    >
      {VirtualizedList}
    </StyledFriendList>
  );
};

export type { FriendListProps };