import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom'; // @version 6.11.2

// Internal imports
import SocialLayout from '../layouts/SocialLayout';
import SocialHub from '../components/social/SocialHub';
import { useFleet } from '../hooks/useFleet';

// Interface definitions
interface SocialPageProps {
  className?: string;
  hardwareToken: string;
  powerMode: 'performance' | 'balanced' | 'powersave';
  networkQuality: NetworkQualityMetrics;
}

interface NetworkQualityMetrics {
  latency: number;
  bandwidth: number;
  packetsLost: number;
  connectionScore: number;
}

/**
 * Social page component with enhanced security and performance features
 * Implements proximity-based discovery, fleet management, and real-time communication
 */
const Social: React.FC<SocialPageProps> = React.memo(({
  className,
  hardwareToken,
  powerMode,
  networkQuality
}) => {
  // Navigation and route hooks
  const navigate = useNavigate();
  const { fleetId } = useParams<{ fleetId: string }>();

  // Fleet management hook with CRDT support
  const {
    currentFleet,
    fleetMembers,
    networkStats,
    createFleet,
    joinFleet,
    validateHardwareToken,
    monitorNetworkQuality
  } = useFleet();

  // Local state management
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  // Memoized privacy settings
  const privacySettings = useMemo(() => ({
    shareLocation: true,
    shareScanData: true,
    dataRetentionDays: 30,
    gdprConsent: true
  }), []);

  /**
   * Handles secure fleet joining with hardware validation
   */
  const handleFleetJoin = useCallback(async (
    targetFleetId: string,
    token: string,
    quality: NetworkQualityMetrics
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      // Validate hardware security token
      const isValidToken = await validateHardwareToken(token);
      if (!isValidToken) {
        throw new Error('Hardware security validation failed');
      }

      // Check network quality requirements
      if (quality.latency > 50 || quality.connectionScore < 0.7) {
        throw new Error('Network quality insufficient for fleet operations');
      }

      // Attempt to join fleet
      const joined = await joinFleet(targetFleetId);
      if (!joined) {
        throw new Error('Failed to join fleet');
      }

      // Initialize network monitoring
      monitorNetworkQuality();

    } catch (error) {
      setError(error instanceof Error ? error : new Error('Fleet join failed'));
      console.error('Fleet join error:', error);
    } finally {
      setIsLoading(false);
    }
  }, [validateHardwareToken, joinFleet, monitorNetworkQuality]);

  // Auto-join fleet if ID is provided
  useEffect(() => {
    if (fleetId && !currentFleet && !isLoading) {
      handleFleetJoin(fleetId, hardwareToken, networkQuality);
    }
  }, [fleetId, currentFleet, hardwareToken, networkQuality, handleFleetJoin, isLoading]);

  // Update page title and metadata
  useEffect(() => {
    document.title = currentFleet 
      ? `Fleet: ${currentFleet.name} - TALD UNIA Social`
      : 'TALD UNIA Social';
  }, [currentFleet]);

  // Monitor and adapt to network quality changes
  useEffect(() => {
    if (currentFleet) {
      monitorNetworkQuality();
    }
  }, [currentFleet, networkQuality, monitorNetworkQuality]);

  // Power-aware optimizations
  useEffect(() => {
    if (powerMode === 'powersave') {
      // Reduce update frequency and disable animations
      document.documentElement.style.setProperty('--animation-duration', '500ms');
    } else {
      document.documentElement.style.setProperty('--animation-duration', '300ms');
    }
  }, [powerMode]);

  return (
    <SocialLayout
      className={className}
      powerMode={powerMode}
      hdrEnabled={true}
      fleetSize={fleetMembers.size}
      networkQuality={networkStats.averageLatency < 50 ? 'good' : 'fair'}
    >
      <SocialHub
        fleetId={currentFleet?.id}
        securityContext="social"
        privacySettings={privacySettings}
        region="auto"
        powerMode={powerMode}
      />
    </SocialLayout>
  );
});

// Display name for debugging
Social.displayName = 'Social';

export default Social;