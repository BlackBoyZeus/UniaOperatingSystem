import React, { useCallback, useEffect, useState } from 'react';
import styled from 'styled-components';
import { useLocation } from 'react-router-dom';
import { useMediaQuery } from 'react-responsive';
import usePerformanceMonitor from '@performance-monitor/react';
import usePowerMode from '@power-mode/react';

import Header from '../components/layout/Header';
import Sidebar from '../components/layout/Sidebar';
import { UI_CONSTANTS } from '../../constants/ui.constants';
import useAuth from '../../hooks/useAuth';
import useFleet from '../../hooks/useFleet';

// Props interface for the DashboardLayout component
interface DashboardLayoutProps {
  children: React.ReactNode;
  className?: string;
}

// GPU-accelerated layout container with HDR support
const LayoutContainer = styled.div<{
  powerMode: string;
  colorMode: 'srgb' | 'display-p3';
}>`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: ${props => props.theme.colors.background.primary};
  color-space: ${props => props.colorMode};
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform;
  contain: content;
  
  /* Power-aware transitions */
  transition: transform ${props => 
    props.powerMode === 'performance' 
      ? UI_CONSTANTS.ANIMATION.DURATION.FAST 
      : UI_CONSTANTS.ANIMATION.DURATION.NORMAL}ms 
    ${UI_CONSTANTS.ANIMATION.EASING.DEFAULT};
`;

const MainContent = styled.main<{ sidebarOpen: boolean }>`
  margin-left: ${props => props.sidebarOpen ? '280px' : '0'};
  margin-top: 64px;
  padding: ${UI_CONSTANTS.SPACING.L}px;
  flex: 1;
  transition: margin-left ${UI_CONSTANTS.ANIMATION.DURATION.NORMAL}ms 
    ${UI_CONSTANTS.ANIMATION.EASING.DEFAULT};
  
  /* Content optimization */
  contain: paint layout;
  will-change: margin-left;
`;

const DashboardLayout: React.FC<DashboardLayoutProps> = React.memo(({ 
  children, 
  className 
}) => {
  // State management
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [colorMode, setColorMode] = useState<'srgb' | 'display-p3'>('srgb');

  // Hooks
  const location = useLocation();
  const { user, monitorSecurityEvents } = useAuth();
  const { currentFleet, networkStats } = useFleet();
  const { startMonitoring, metrics } = usePerformanceMonitor();
  const { powerMode, batteryLevel } = usePowerMode();
  
  // Media queries for responsive and HDR support
  const isWideScreen = useMediaQuery({ minWidth: 1024 });
  const supportsHDR = useMediaQuery({ query: '(dynamic-range: high)' });

  // Initialize performance monitoring
  useEffect(() => {
    startMonitoring({
      maxLatency: 16, // Target 60fps
      sampleInterval: 1000,
      reportCallback: (metrics) => {
        if (metrics.averageLatency > 16) {
          console.warn('Performance degradation detected');
        }
      }
    });
  }, [startMonitoring]);

  // Update color mode based on HDR support
  useEffect(() => {
    setColorMode(supportsHDR ? 'display-p3' : 'srgb');
  }, [supportsHDR]);

  // Security monitoring
  useEffect(() => {
    if (user) {
      const securityInterval = setInterval(() => {
        monitorSecurityEvents().catch(console.error);
      }, 30000);
      return () => clearInterval(securityInterval);
    }
  }, [user, monitorSecurityEvents]);

  // Sidebar toggle handler with power awareness
  const handleSidebarToggle = useCallback(() => {
    setSidebarOpen(prev => !prev);
  }, []);

  // Auto-close sidebar on navigation for mobile
  useEffect(() => {
    if (!isWideScreen && sidebarOpen) {
      setSidebarOpen(false);
    }
  }, [location.pathname, isWideScreen]);

  return (
    <LayoutContainer
      className={className}
      powerMode={powerMode}
      colorMode={colorMode}
    >
      <Header
        transparent={false}
        powerMode={powerMode}
        colorSpace={colorMode}
        onMenuClick={handleSidebarToggle}
        fleetStatus={currentFleet?.status}
        networkStats={networkStats}
        batteryLevel={batteryLevel}
      />
      
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        hdrEnabled={colorMode === 'display-p3'}
        powerMode={powerMode === 'POWER_SAVER' ? 'power-save' : 
                  powerMode === 'HIGH_PERFORMANCE' ? 'performance' : 
                  'balanced'}
      />
      
      <MainContent 
        sidebarOpen={sidebarOpen && isWideScreen}
        role="main"
      >
        {children}
      </MainContent>
    </LayoutContainer>
  );
});

DashboardLayout.displayName = 'DashboardLayout';

export type { DashboardLayoutProps };
export default DashboardLayout;