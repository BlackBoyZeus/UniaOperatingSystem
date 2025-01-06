import React, { useCallback, useEffect, useState } from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';
import Icon, { IconProps } from '../common/Icon';
import Button from '../common/Button';
import useAuth from '../../hooks/useAuth';
import { UI_CONSTANTS } from '../../constants/ui.constants';
import { UserRoleType } from '../../interfaces/user.interface';

// Navigation variants for different contexts
type NavigationVariant = 'default' | 'game' | 'fleet';
type PowerMode = 'normal' | 'battery' | 'performance';

interface NavigationProps {
  className?: string;
  variant?: NavigationVariant;
  powerMode?: PowerMode;
}

// Core navigation items with role-based access control
const NAV_ITEMS = [
  {
    icon: 'MENU',
    label: 'Play',
    path: '/game',
    requiredRole: UserRoleType.USER,
    requiresHardwareAuth: true
  },
  {
    icon: 'STORE',
    label: 'Store',
    path: '/store',
    requiredRole: UserRoleType.USER
  },
  {
    icon: 'PROFILE',
    label: 'Social',
    path: '/social',
    requiredRole: UserRoleType.USER,
    fleetStatus: true
  },
  {
    icon: 'SETTINGS',
    label: 'Settings',
    path: '/settings'
  }
] as const;

// GPU-accelerated styled components with HDR support
const StyledNav = styled.nav<{
  variant: NavigationVariant;
  powerMode: PowerMode;
}>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${UI_CONSTANTS.SPACING?.CONTAINER_PADDING || '16px'};
  background: ${({ theme, variant }) => 
    variant === 'game' 
      ? 'color(display-p3 0.1 0.1 0.15 / 0.8)' 
      : theme.palette.background};
  color: ${({ theme }) => theme.palette.HDR_COLORS?.text || '#FFFFFF'};
  backdrop-filter: blur(8px);
  z-index: ${UI_CONSTANTS.Z_INDEX?.HEADER || 100};
  
  /* GPU acceleration optimizations */
  will-change: transform;
  transform: translate3d(0, 0, 0);
  transition: transform ${({ powerMode }) => 
    powerMode === 'battery' 
      ? UI_CONSTANTS.ANIMATION?.DURATION?.SLOW 
      : UI_CONSTANTS.ANIMATION?.DURATION?.FAST}ms 
    ${UI_CONSTANTS.ANIMATION?.EASING?.DEFAULT};

  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const NavGroup = styled.div`
  display: flex;
  gap: 16px;
  align-items: center;
`;

const NavButton = styled(Button)<{ isActive?: boolean }>`
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: ${({ isActive, theme }) => 
    isActive 
      ? 'color(display-p3 0.2 0.2 0.25 / 0.9)' 
      : 'transparent'};
  border-radius: 8px;

  &:hover {
    background: color(display-p3 0.2 0.2 0.25 / 0.7);
  }
`;

const FleetStatus = styled.div<{ isActive: boolean }>`
  position: absolute;
  top: 4px;
  right: 4px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ isActive }) => 
    isActive 
      ? 'color(display-p3 0 1 0)' 
      : 'color(display-p3 1 0 0)'};
  box-shadow: 0 0 8px ${({ isActive }) => 
    isActive 
      ? 'color(display-p3 0 1 0 / 0.5)' 
      : 'color(display-p3 1 0 0 / 0.5)'};
`;

export const Navigation: React.FC<NavigationProps> = ({
  className,
  variant = 'default',
  powerMode = 'normal'
}) => {
  const navigate = useNavigate();
  const { user, hasRole, isAuthenticated } = useAuth();
  const [activeFleet, setActiveFleet] = useState(false);

  // Enhanced navigation handler with security validation
  const handleNavigation = useCallback(async (
    path: string,
    requiredRole?: UserRoleType,
    requiresHardwareAuth?: boolean
  ) => {
    try {
      // Role-based access control
      if (requiredRole && !hasRole(requiredRole)) {
        throw new Error('Insufficient permissions');
      }

      // Hardware token validation if required
      if (requiresHardwareAuth && !user?.deviceCapabilities?.hardwareSecurityLevel) {
        throw new Error('Hardware authentication required');
      }

      navigate(path);
    } catch (error) {
      console.error('Navigation error:', error);
      // Handle navigation error (could show a notification)
    }
  }, [navigate, hasRole, user]);

  // Monitor fleet status
  useEffect(() => {
    if (!isAuthenticated) return;

    const checkFleetStatus = async () => {
      try {
        // Simulated fleet status check - replace with actual implementation
        setActiveFleet(user?.fleetId != null);
      } catch (error) {
        console.error('Fleet status check failed:', error);
      }
    };

    const intervalId = setInterval(checkFleetStatus, 5000);
    checkFleetStatus();

    return () => clearInterval(intervalId);
  }, [isAuthenticated, user]);

  return (
    <StyledNav 
      className={className}
      variant={variant}
      powerMode={powerMode}
      role="navigation"
      aria-label="Main navigation"
    >
      <NavGroup>
        {NAV_ITEMS.map(item => (
          <NavButton
            key={item.path}
            onClick={() => handleNavigation(
              item.path,
              item.requiredRole,
              item.requiresHardwareAuth
            )}
            variant="transparent"
            powerSaveAware={powerMode === 'battery'}
            disabled={item.requiredRole && !hasRole(item.requiredRole)}
            aria-label={item.label}
          >
            <Icon 
              name={UI_CONSTANTS.ICONS?.[item.icon] || item.icon}
              size={24}
              powerMode={powerMode === 'battery' ? 'POWER_SAVER' : 'BALANCED'}
            />
            {item.label}
            {item.fleetStatus && (
              <FleetStatus 
                isActive={activeFleet}
                aria-label={`Fleet status: ${activeFleet ? 'Active' : 'Inactive'}`}
              />
            )}
          </NavButton>
        ))}
      </NavGroup>
    </StyledNav>
  );
};

export default React.memo(Navigation);