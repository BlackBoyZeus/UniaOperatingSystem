import React, { useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { Card, CardProps } from '../common/Card';
import { IUserProfile, UserStatusType } from '../../interfaces/user.interface';
import { useAuth } from '../../hooks/useAuth';

/**
 * @external react v18.2.0
 * @external @emotion/styled v11.11.0
 */

interface ProfileCardProps {
  profile: IUserProfile;
  onClick?: (profile: IUserProfile) => void;
  className?: string;
  interactive?: boolean;
  showFleetStatus?: boolean;
  showScanningStatus?: boolean;
  animationEnabled?: boolean;
}

const StyledProfileCard = styled(Card)<{ isInteractive: boolean }>`
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  color-scheme: dark light;
  transition: transform 100ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform;
  backface-visibility: hidden;
  transform: translateZ(0);
  touch-action: manipulation;
  color-gamut: p3;
  color-rendering: optimizeSpeed;

  ${({ isInteractive }) =>
    isInteractive &&
    `
    cursor: pointer;
    &:hover {
      transform: translateZ(0) scale3d(1.02, 1.02, 1);
    }
    &:active {
      transform: translateZ(0) scale3d(0.98, 0.98, 1);
    }
  `}
`;

const Avatar = styled.img`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
  content-visibility: auto;
`;

const ProfileInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const DisplayName = styled.h3`
  margin: 0;
  font-family: var(--font-family-gaming);
  font-weight: var(--font-weight-medium);
  color: color(display-p3 1 1 1);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const StatusIndicator = styled.div<{ color: string }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-left: 8px;
  background-color: ${({ color }) => color};
  transition: background-color 150ms ease-out;
  will-change: background-color;
`;

const StatusContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: color(display-p3 0.8 0.8 0.8);
`;

const FleetBadge = styled.span`
  padding: 2px 8px;
  border-radius: 12px;
  background: color(display-p3 0.486 0.302 1 / 0.2);
  color: color(display-p3 0.486 0.302 1);
  font-size: 12px;
  font-weight: var(--font-weight-medium);
`;

export const ProfileCard: React.FC<ProfileCardProps> = ({
  profile,
  onClick,
  className,
  interactive = false,
  showFleetStatus = true,
  showScanningStatus = true,
  animationEnabled = true,
}) => {
  const { hasRole } = useAuth();

  const handleProfileClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!interactive || !onClick) return;
      event.stopPropagation();
      onClick(profile);
    },
    [interactive, onClick, profile]
  );

  const statusColor = useMemo(() => {
    if (showScanningStatus && profile.isScanning) {
      return 'color(display-p3 0 0.898 1)'; // Scanning blue
    }
    
    switch (profile.status) {
      case UserStatusType.ONLINE:
        return 'color(display-p3 0.2 0.8 0.2)'; // Online green
      case UserStatusType.IN_GAME:
        return 'color(display-p3 0.486 0.302 1)'; // In-game purple
      case UserStatusType.IN_FLEET:
        return 'color(display-p3 1 0.5 0)'; // Fleet orange
      case UserStatusType.SCANNING:
        return 'color(display-p3 0 0.898 1)'; // Scanning blue
      default:
        return 'color(display-p3 0.5 0.5 0.5)'; // Offline gray
    }
  }, [profile.status, profile.isScanning, showScanningStatus]);

  return (
    <StyledProfileCard
      variant="elevated"
      isInteractive={interactive}
      onClick={handleProfileClick}
      className={className}
      hdrMode="auto"
      powerSaveMode={!animationEnabled}
      data-testid="profile-card"
    >
      <Avatar
        src={profile.avatar}
        alt={profile.displayName}
        loading="lazy"
        data-testid="profile-avatar"
      />
      <ProfileInfo>
        <DisplayName data-testid="profile-name">
          {profile.displayName}
        </DisplayName>
        <StatusContainer>
          <StatusIndicator color={statusColor} data-testid="status-indicator" />
          {profile.status}
          {showFleetStatus && profile.fleetId && (
            <FleetBadge data-testid="fleet-badge">
              Fleet Member
            </FleetBadge>
          )}
          {hasRole('FLEET_LEADER') && profile.fleetId && (
            <FleetBadge data-testid="fleet-leader-badge">
              Fleet Leader
            </FleetBadge>
          )}
        </StatusContainer>
      </ProfileInfo>
    </StyledProfileCard>
  );
};

export type { ProfileCardProps };