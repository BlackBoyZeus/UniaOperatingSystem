import React, { useMemo } from 'react';
import styled from '@emotion/styled';
import { Card, CardProps } from '../common/Card';
import { IUserProfile, IUser, DeviceCapabilityType } from '../../interfaces/user.interface';

/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 */

export interface ProfileStatsProps {
  user: IUser;
  profile: IUserProfile;
  className?: string;
  powerSaveMode?: boolean;
  hdrEnabled?: boolean;
  fleetStatus?: string;
}

const StyledStatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  padding: 16px;
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;
  contain: layout style paint;
  color-scheme: dark light;
  color-gamut: p3;
`;

const StyledStatItem = styled.div<{ powerSaveMode?: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border-radius: 8px;
  background: color(display-p3 0 0 0 / 0.05);
  transition: transform 100ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform;
  touch-action: manipulation;
  contain: content;

  ${({ powerSaveMode }) =>
    !powerSaveMode &&
    `
    &:hover {
      transform: scale(1.02);
      transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    }
  `}

  ${({ powerSaveMode }) =>
    powerSaveMode &&
    `
    transition: none;
    transform: none;
    animation: none;
  `}
`;

const StatLabel = styled.span`
  font-family: var(--font-family-gaming);
  font-size: 14px;
  color: color(display-p3 0.7 0.7 0.7);
  font-feature-settings: var(--font-feature-gaming);
`;

const StatValue = styled.span<{ hdrEnabled?: boolean }>`
  font-family: var(--font-family-gaming);
  font-size: 18px;
  font-weight: var(--font-weight-medium);
  color: ${({ hdrEnabled }) =>
    hdrEnabled
      ? 'var(--color-primary-hdr)'
      : 'var(--color-primary)'};
`;

const formatDeviceCapabilities = (capabilities: DeviceCapabilityType): Record<string, string> => {
  return {
    lidar: `LiDAR: ${capabilities.lidarSupported ? `${capabilities.scanningResolution}cm` : 'Not Supported'}`,
    network: `Mesh Network: ${capabilities.meshNetworkSupported ? `${capabilities.maxFleetSize} devices` : 'Not Supported'}`,
    vulkan: `Vulkan: ${capabilities.vulkanVersion}`,
    security: `Security: ${capabilities.hardwareSecurityLevel}`,
  };
};

const formatLastActive = (lastActive: Date, fleetStatus?: string): string => {
  const now = new Date();
  const diffMinutes = Math.floor((now.getTime() - lastActive.getTime()) / 60000);

  if (fleetStatus === 'IN_FLEET' || fleetStatus === 'IN_GAME') {
    return 'Currently Active';
  }

  if (diffMinutes < 1) {
    return 'Just now';
  } else if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  } else if (diffMinutes < 1440) {
    return `${Math.floor(diffMinutes / 60)}h ago`;
  }
  return `${Math.floor(diffMinutes / 1440)}d ago`;
};

export const ProfileStats: React.FC<ProfileStatsProps> = ({
  user,
  profile,
  className,
  powerSaveMode = false,
  hdrEnabled = false,
  fleetStatus,
}) => {
  const deviceStats = useMemo(() => formatDeviceCapabilities(user.deviceCapabilities), [user.deviceCapabilities]);
  const lastActiveFormatted = useMemo(() => formatLastActive(user.lastActive, fleetStatus), [user.lastActive, fleetStatus]);

  return (
    <Card
      variant="elevated"
      className={className}
      powerSaveMode={powerSaveMode}
      hdrMode={hdrEnabled ? 'enabled' : 'disabled'}
    >
      <StyledStatsGrid>
        <StyledStatItem powerSaveMode={powerSaveMode}>
          <StatLabel>Status</StatLabel>
          <StatValue hdrEnabled={hdrEnabled}>{profile.status}</StatValue>
        </StyledStatItem>

        <StyledStatItem powerSaveMode={powerSaveMode}>
          <StatLabel>Last Active</StatLabel>
          <StatValue hdrEnabled={hdrEnabled}>{lastActiveFormatted}</StatValue>
        </StyledStatItem>

        {Object.entries(deviceStats).map(([key, value]) => (
          <StyledStatItem key={key} powerSaveMode={powerSaveMode}>
            <StatLabel>{key.charAt(0).toUpperCase() + key.slice(1)}</StatLabel>
            <StatValue hdrEnabled={hdrEnabled}>{value}</StatValue>
          </StyledStatItem>
        ))}

        <StyledStatItem powerSaveMode={powerSaveMode}>
          <StatLabel>Active Scans</StatLabel>
          <StatValue hdrEnabled={hdrEnabled}>{profile.activeScans}</StatValue>
        </StyledStatItem>

        <StyledStatItem powerSaveMode={powerSaveMode}>
          <StatLabel>Fleet Status</StatLabel>
          <StatValue hdrEnabled={hdrEnabled}>
            {profile.fleetId ? `Fleet ${profile.fleetId.slice(0, 8)}...` : 'No Fleet'}
          </StatValue>
        </StyledStatItem>
      </StyledStatsGrid>
    </Card>
  );
};