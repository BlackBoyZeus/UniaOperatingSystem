import React, { useCallback, useEffect, useMemo } from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';

import Icon, { IconProps } from '../common/Icon';
import Button from '../common/Button';
import { UI_CONSTANTS } from '../../constants/ui.constants';
import useAuth from '../../hooks/useAuth';

// Power optimization modes
type PowerMode = 'performance' | 'balanced' | 'power-save';

// Sidebar item configuration with security levels
interface SidebarItem {
  icon: string;
  label: string;
  path: string;
  requiredRole?: string;
  securityLevel: 'standard' | 'elevated' | 'high';
}

// Props interface for the Sidebar component
interface SidebarProps {
  isOpen: boolean;
  onClose?: () => void;
  className?: string;
  hdrEnabled?: boolean;
  powerMode?: PowerMode;
}

// GPU-accelerated styled sidebar container
const StyledSidebar = styled.aside<{
  isOpen: boolean;
  hdrEnabled?: boolean;
  powerMode: PowerMode;
}>`
  position: fixed;
  top: 64px;
  left: 0;
  width: 280px;
  height: calc(100vh - 64px);
  background: ${props => props.hdrEnabled 
    ? 'color(display-p3 0.15 0.15 0.2)' 
    : 'rgb(38, 38, 51)'};
  z-index: ${UI_CONSTANTS.Z_INDEX.SIDEBAR};
  transform: translateX(${props => props.isOpen ? '0' : '-100%'});
  will-change: transform;
  transition: transform ${props => 
    props.powerMode === 'performance' 
      ? '150ms' 
      : props.powerMode === 'power-save' 
        ? '500ms' 
        : '300ms'
  } cubic-bezier(0.4, 0, 0.2, 1);
  backface-visibility: hidden;
  transform-style: preserve-3d;
  contain: layout paint size;
  box-shadow: ${props => props.hdrEnabled 
    ? '0 0 20px rgba(0, 0, 0, 0.3)' 
    : '2px 0 8px rgba(0, 0, 0, 0.2)'};
`;

// Navigation item container
const NavItem = styled(Button)<{ securityLevel: string }>`
  width: 100%;
  padding: ${UI_CONSTANTS.SPACING.M}px;
  margin: ${UI_CONSTANTS.SPACING.XS}px 0;
  display: flex;
  align-items: center;
  gap: ${UI_CONSTANTS.SPACING.S}px;
  border-left: 3px solid transparent;
  border-left-color: ${props => 
    props.securityLevel === 'high' 
      ? 'var(--color-accent)' 
      : props.securityLevel === 'elevated'
        ? 'var(--color-secondary)'
        : 'transparent'
  };
`;

// Core navigation items with security levels
const SIDEBAR_ITEMS: SidebarItem[] = [
  {
    icon: 'PLAY',
    label: 'Play',
    path: '/game',
    securityLevel: 'standard'
  },
  {
    icon: 'STORE',
    label: 'Store',
    path: '/store',
    securityLevel: 'standard'
  },
  {
    icon: 'SOCIAL',
    label: 'Social Hub',
    path: '/social',
    securityLevel: 'elevated'
  },
  {
    icon: 'SETTINGS',
    label: 'Settings',
    path: '/settings',
    requiredRole: 'user',
    securityLevel: 'high'
  }
];

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  className,
  hdrEnabled = false,
  powerMode = 'balanced'
}) => {
  const navigate = useNavigate();
  const { user, hasRole, monitorSecurityEvents } = useAuth();

  // Security monitoring effect
  useEffect(() => {
    if (isOpen && user) {
      monitorSecurityEvents().catch(console.error);
    }
  }, [isOpen, user, monitorSecurityEvents]);

  // Enhanced navigation handler with security validation
  const handleNavigation = useCallback((path: string, requiredRole?: string) => {
    if (requiredRole && !hasRole(requiredRole)) {
      console.warn('Access denied: insufficient permissions');
      return;
    }

    navigate(path);
    onClose?.();
  }, [navigate, hasRole, onClose]);

  // Memoized navigation items
  const navigationItems = useMemo(() => 
    SIDEBAR_ITEMS.map(item => (
      <NavItem
        key={item.path}
        variant="secondary"
        powerSaveAware={powerMode === 'power-save'}
        hdrMode={hdrEnabled ? 'enabled' : 'disabled'}
        securityLevel={item.securityLevel}
        onClick={() => handleNavigation(item.path, item.requiredRole)}
      >
        <Icon 
          name={item.icon}
          size={24}
          powerMode={powerMode === 'power-save' ? 'POWER_SAVER' : 'BALANCED'}
        />
        {item.label}
      </NavItem>
    )),
    [handleNavigation, hdrEnabled, powerMode]
  );

  return (
    <StyledSidebar
      isOpen={isOpen}
      hdrEnabled={hdrEnabled}
      powerMode={powerMode}
      className={className}
      aria-hidden={!isOpen}
      role="navigation"
    >
      {navigationItems}
    </StyledSidebar>
  );
};

export type { SidebarProps };
export default React.memo(Sidebar);