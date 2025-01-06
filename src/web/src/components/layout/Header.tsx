import React, { useCallback, useEffect, useState } from 'react';
import styled from '@emotion/styled';
import { useFrame } from '@react-three/fiber';
import Icon, { IconProps } from '../common/Icon';
import Button from '../common/Button';
import Dropdown from '../common/Dropdown';
import { useAuth } from '../../hooks/useAuth';
import { useFleet } from '../../hooks/useFleet';

// Types
type PowerMode = 'HIGH_PERFORMANCE' | 'BALANCED' | 'POWER_SAVER';
type ColorSpace = 'srgb' | 'display-p3';

interface HeaderProps {
  className?: string;
  transparent?: boolean;
  powerMode?: PowerMode;
  colorSpace?: ColorSpace;
}

// Styled Components with HDR and power-aware optimizations
const StyledHeader = styled.header<{
  transparent: boolean;
  powerMode: PowerMode;
  colorSpace: ColorSpace;
}>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: ${props => props.transparent ? 'transparent' : 'var(--color-surface)'};
  backdrop-filter: blur(8px);
  z-index: var(--z-index-header);
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform, background-color;
  
  /* HDR color space support */
  color-space: ${props => props.colorSpace};
  
  /* Power-aware animations */
  transition: all ${props => props.powerMode === 'POWER_SAVER' ? '500ms' : '300ms'} 
    cubic-bezier(0.4, 0, 0.2, 1);
  
  /* Touch optimizations */
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const NavigationContainer = styled.nav`
  display: flex;
  align-items: center;
  gap: 24px;
`;

const FleetStatus = styled.div<{ status: string }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px;
  border-radius: 4px;
  background: ${props => props.status === 'ACTIVE' ? 
    'color(display-p3 0 0.8 0 / 0.2)' : 
    'color(display-p3 0.8 0 0 / 0.2)'};
  color: ${props => props.status === 'ACTIVE' ? 
    'color(display-p3 0 1 0)' : 
    'color(display-p3 1 0 0)'};
  font-size: 14px;
  font-weight: 500;
  
  /* GPU acceleration */
  transform: translateZ(0);
  will-change: background-color, color;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
`;

const UserContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

const Header: React.FC<HeaderProps> = ({
  className,
  transparent = false,
  powerMode = 'BALANCED',
  colorSpace = 'display-p3'
}) => {
  const { user, logout } = useAuth();
  const { currentFleet, fleetMembers, networkStats } = useFleet();
  const [frameTime, setFrameTime] = useState(0);

  // Monitor frame timing for performance optimization
  useFrame((state, delta) => {
    setFrameTime(delta * 1000);
  });

  // Enhanced logout handler with hardware token cleanup
  const handleLogout = useCallback(async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  }, [logout]);

  // Update power mode based on frame timing
  useEffect(() => {
    if (frameTime > 16.66) { // Below 60fps
      document.documentElement.style.setProperty('--animation-duration', '500ms');
    } else {
      document.documentElement.style.setProperty('--animation-duration', '300ms');
    }
  }, [frameTime]);

  return (
    <StyledHeader
      className={className}
      transparent={transparent}
      powerMode={powerMode}
      colorSpace={colorSpace}
    >
      <NavigationContainer>
        <Button
          variant="primary"
          size="medium"
          enableHaptic
          hdrMode="auto"
          powerSaveAware
        >
          <Icon name="menu" size={24} powerMode={powerMode} />
        </Button>

        {currentFleet && (
          <FleetStatus status={currentFleet.status}>
            <Icon 
              name={currentFleet.status === 'ACTIVE' ? 'fleet-active' : 'fleet-inactive'} 
              size={16}
              animate={currentFleet.status === 'ACTIVE'}
              powerMode={powerMode}
            />
            Fleet: {fleetMembers.size}/{currentFleet.maxDevices}
            {networkStats.averageLatency > 0 && (
              <span>({Math.round(networkStats.averageLatency)}ms)</span>
            )}
          </FleetStatus>
        )}
      </NavigationContainer>

      <UserContainer>
        {user ? (
          <>
            <Dropdown
              options={[
                { value: 'profile', label: 'Profile' },
                { value: 'settings', label: 'Settings' },
                { value: 'logout', label: 'Logout' }
              ]}
              value=""
              onChange={(value) => {
                if (value === 'logout') {
                  handleLogout();
                }
              }}
              powerMode={powerMode}
              hdrEnabled={colorSpace === 'display-p3'}
              width={150}
            />
            <Icon 
              name="user" 
              size={24}
              powerMode={powerMode}
            />
          </>
        ) : (
          <Button
            variant="primary"
            size="medium"
            enableHaptic
            hdrMode="auto"
            powerSaveAware
          >
            Login
          </Button>
        )}
      </UserContainer>
    </StyledHeader>
  );
};

export default React.memo(Header);