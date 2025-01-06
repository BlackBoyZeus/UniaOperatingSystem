import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import debounce from 'lodash/debounce';
import { Button } from '../common/Button';
import CustomSlider from '../common/Slider';
import { useAuth } from '../../hooks/useAuth';
import { ANIMATION_TIMINGS, PERFORMANCE_THRESHOLDS, UI_INTERACTIONS } from '../../constants/ui.constants';

// Enhanced props interface for SystemSettings component
interface SystemSettingsProps {
  onSettingsChange: (settings: SystemSettingsType, validation: ValidationResult) => Promise<void>;
  className?: string;
  securityLevel?: SecurityLevelType;
  powerProfile?: PowerProfileType;
}

// Enhanced type definitions
interface SystemSettingsType {
  lidarQuality: number;
  meshNetworkLatency: number;
  powerMode: 'performance' | 'balanced' | 'powersaver';
  vulkanOptimization: boolean;
  hardwareSecurity: HardwareSecurityLevel;
  gpuAcceleration: GPUAccelerationType;
}

interface ValidationResult {
  isValid: boolean;
  hardwareSupported: boolean;
  securityCompliant: boolean;
  powerImpact: number;
}

type SecurityLevelType = 'high' | 'medium' | 'low';
type PowerProfileType = 'performance' | 'balanced' | 'powersaver';
type HardwareSecurityLevel = 'tpm' | 'hardware' | 'software';
type GPUAccelerationType = 'full' | 'partial' | 'disabled';

// GPU-accelerated styled components
const SettingsContainer = styled.div`
  padding: calc(var(--spacing-unit) * 3);
  border-radius: calc(var(--spacing-unit));
  background: var(--color-surface);
  color: var(--color-primary);
  transform: translateZ(0);
  will-change: transform;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  content-visibility: auto;
  contain: content;

  @media (dynamic-range: high) {
    background: color(display-p3 var(--color-surface));
    color: color(display-p3 var(--color-primary-hdr));
  }
`;

const SettingGroup = styled.div`
  margin-bottom: calc(var(--spacing-unit) * 3);
  opacity: 1;
  transform: translateZ(0);
  transition: opacity 150ms ease-in-out;

  &:disabled {
    opacity: 0.5;
    pointer-events: none;
  }
`;

const SettingLabel = styled.label`
  display: block;
  margin-bottom: var(--spacing-unit);
  font-family: var(--font-family-gaming);
  font-weight: var(--font-weight-medium);
  color: var(--color-primary);
`;

// Enhanced SystemSettings component
export const SystemSettings: React.FC<SystemSettingsProps> = ({
  onSettingsChange,
  className,
  securityLevel = 'high',
  powerProfile = 'balanced'
}) => {
  const { user, hasRole, validateHardwareSecurity } = useAuth();
  
  // State management with hardware validation
  const [settings, setSettings] = useState<SystemSettingsType>({
    lidarQuality: 80,
    meshNetworkLatency: 50,
    powerMode: 'balanced',
    vulkanOptimization: true,
    hardwareSecurity: 'tpm',
    gpuAcceleration: 'full'
  });

  // Performance monitoring state
  const [performanceMetrics, setPerformanceMetrics] = useState({
    frameTime: 0,
    powerUsage: 0,
    gpuUtilization: 0
  });

  // Memoized hardware capabilities
  const deviceCapabilities = useMemo(() => ({
    maxLidarQuality: user?.deviceCapabilities?.scanningResolution || 100,
    minLatency: 30,
    maxLatency: 100,
    supportsVulkan: user?.deviceCapabilities?.vulkanVersion !== undefined,
    securityLevel: user?.deviceCapabilities?.hardwareSecurityLevel || 'software'
  }), [user]);

  // Debounced settings update with hardware validation
  const updateSettings = useCallback(debounce(async (
    newSettings: Partial<SystemSettingsType>
  ) => {
    const updatedSettings = { ...settings, ...newSettings };
    
    // Validate hardware capabilities
    const validation: ValidationResult = {
      isValid: true,
      hardwareSupported: true,
      securityCompliant: true,
      powerImpact: 0
    };

    // Hardware security validation
    if (securityLevel === 'high') {
      const securityValidation = await validateHardwareSecurity(updatedSettings.hardwareSecurity);
      validation.securityCompliant = securityValidation;
    }

    // Performance impact calculation
    validation.powerImpact = calculatePowerImpact(updatedSettings);

    // Update if valid
    if (validation.isValid && validation.securityCompliant) {
      setSettings(updatedSettings);
      await onSettingsChange(updatedSettings, validation);
    }
  }, ANIMATION_TIMINGS.MIN_FRAME_TIME), [settings, securityLevel, validateHardwareSecurity]);

  // Hardware-validated LiDAR quality handler
  const handleLidarQualityChange = useCallback((value: number) => {
    if (value <= deviceCapabilities.maxLidarQuality) {
      updateSettings({ lidarQuality: value });
    }
  }, [deviceCapabilities.maxLidarQuality, updateSettings]);

  // Power-aware mode handler
  const handlePowerModeChange = useCallback((mode: PowerProfileType) => {
    const gpuAcceleration = mode === 'performance' ? 'full' : mode === 'balanced' ? 'partial' : 'disabled';
    updateSettings({ 
      powerMode: mode,
      gpuAcceleration,
      vulkanOptimization: mode !== 'powersaver'
    });
  }, [updateSettings]);

  // Performance monitoring
  useEffect(() => {
    const monitorPerformance = () => {
      const frameTime = performance.now();
      // Monitor real performance metrics
      setPerformanceMetrics(prev => ({
        frameTime: frameTime - prev.frameTime,
        powerUsage: calculatePowerUsage(settings),
        gpuUtilization: calculateGPUUtilization(settings)
      }));
      
      requestAnimationFrame(monitorPerformance);
    };

    const frameId = requestAnimationFrame(monitorPerformance);
    return () => cancelAnimationFrame(frameId);
  }, [settings]);

  return (
    <SettingsContainer className={className}>
      <SettingGroup>
        <SettingLabel>LiDAR Quality</SettingLabel>
        <CustomSlider
          value={settings.lidarQuality}
          onChange={handleLidarQualityChange}
          min={0}
          max={deviceCapabilities.maxLidarQuality}
          step={1}
          label="LiDAR Quality"
          powerMode={settings.powerMode}
          criticalSetting={true}
        />
      </SettingGroup>

      <SettingGroup>
        <SettingLabel>Network Latency</SettingLabel>
        <CustomSlider
          value={settings.meshNetworkLatency}
          onChange={(value) => updateSettings({ meshNetworkLatency: value })}
          min={deviceCapabilities.minLatency}
          max={deviceCapabilities.maxLatency}
          step={1}
          label="Network Latency"
          powerMode={settings.powerMode}
        />
      </SettingGroup>

      <SettingGroup>
        <SettingLabel>Power Mode</SettingLabel>
        <Button
          variant={settings.powerMode === 'performance' ? 'primary' : 'secondary'}
          onClick={() => handlePowerModeChange('performance')}
          powerSaveAware={true}
        >
          Performance
        </Button>
        <Button
          variant={settings.powerMode === 'balanced' ? 'primary' : 'secondary'}
          onClick={() => handlePowerModeChange('balanced')}
          powerSaveAware={true}
        >
          Balanced
        </Button>
        <Button
          variant={settings.powerMode === 'powersaver' ? 'primary' : 'secondary'}
          onClick={() => handlePowerModeChange('powersaver')}
          powerSaveAware={true}
        >
          Power Saver
        </Button>
      </SettingGroup>

      <SettingGroup>
        <SettingLabel>Hardware Acceleration</SettingLabel>
        <Button
          variant="secondary"
          onClick={() => updateSettings({ vulkanOptimization: !settings.vulkanOptimization })}
          disabled={!deviceCapabilities.supportsVulkan}
          powerSaveAware={true}
        >
          Vulkan Optimization: {settings.vulkanOptimization ? 'Enabled' : 'Disabled'}
        </Button>
      </SettingGroup>
    </SettingsContainer>
  );
};

// Utility functions
const calculatePowerImpact = (settings: SystemSettingsType): number => {
  let impact = 0;
  impact += settings.lidarQuality * 0.5;
  impact += settings.vulkanOptimization ? 20 : 0;
  impact += settings.powerMode === 'performance' ? 30 : settings.powerMode === 'balanced' ? 15 : 0;
  return Math.min(100, impact);
};

const calculatePowerUsage = (settings: SystemSettingsType): number => {
  return settings.powerMode === 'performance' ? 100 :
         settings.powerMode === 'balanced' ? 70 : 40;
};

const calculateGPUUtilization = (settings: SystemSettingsType): number => {
  return settings.gpuAcceleration === 'full' ? 90 :
         settings.gpuAcceleration === 'partial' ? 60 : 30;
};

export type { SystemSettingsProps, SystemSettingsType };