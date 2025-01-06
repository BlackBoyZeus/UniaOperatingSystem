import React, { useCallback, useMemo } from 'react';
import styled from '@emotion/styled';
import { useTheme } from '@emotion/react';
import { usePerformanceMonitor } from '@performance-monitor/react'; // v1.0.0
import { usePowerAware } from '@power-aware/react'; // v1.0.0
import CustomSlider from '../common/Slider';
import Dropdown from '../common/Dropdown';
import { ANIMATION_TIMINGS, UI_INTERACTIONS, PERFORMANCE_THRESHOLDS } from '../../constants/ui.constants';

// Types
interface IResolution {
  width: number;
  height: number;
  refreshRate: number;
}

type RenderQuality = 'PERFORMANCE' | 'BALANCED' | 'QUALITY';
type PowerMode = keyof typeof ANIMATION_TIMINGS.POWER_MODES;

interface GraphicsSettingsProps {
  resolution: IResolution;
  quality: RenderQuality;
  lidarQuality: number;
  onResolutionChange: (resolution: IResolution) => Promise<void>;
  onQualityChange: (quality: RenderQuality) => Promise<void>;
  onLidarQualityChange: (quality: number) => Promise<void>;
  powerMode: PowerMode;
}

// Styled Components with GPU acceleration and HDR support
const SettingsContainer = styled.div<{ powerMode: PowerMode }>`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  color-gamut: p3;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000;
  will-change: transform;
  
  /* Power-aware animations */
  transition-property: all;
  transition-timing-function: ${ANIMATION_TIMINGS.EASING.DEFAULT};
  transition-duration: ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms;
`;

const SettingGroup = styled.div<{ isActive: boolean; powerMode: PowerMode }>`
  display: flex;
  flex-direction: column;
  gap: 8px;
  opacity: ${props => props.isActive ? 1 : 0.7};
  
  /* Power-aware transitions */
  transition: opacity ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
`;

const Label = styled.label`
  font-family: ${props => props.theme.typography.gaming.family};
  font-weight: ${props => props.theme.typography.gaming.weights.regular};
  color: ${props => props.theme.colorDepth.hdrEnabled ? 
    'color(display-p3 1 1 1)' : 
    props.theme.palette.text.primary};
`;

// Resolution options
const RESOLUTION_OPTIONS = [
  { value: '1920x1080@60', label: '1920x1080 (60Hz)' },
  { value: '2560x1440@60', label: '2560x1440 (60Hz)' },
  { value: '3840x2160@60', label: '3840x2160 (60Hz)' }
];

const QUALITY_OPTIONS = [
  { value: 'PERFORMANCE', label: 'Performance' },
  { value: 'BALANCED', label: 'Balanced' },
  { value: 'QUALITY', label: 'Quality' }
];

const GraphicsSettings: React.FC<GraphicsSettingsProps> = React.memo(({
  resolution,
  quality,
  lidarQuality,
  onResolutionChange,
  onQualityChange,
  onLidarQualityChange,
  powerMode
}) => {
  const theme = useTheme();
  const { trackPerformance } = usePerformanceMonitor();
  const { isPowerConstrained } = usePowerAware();

  // Parse resolution string to IResolution
  const parseResolution = useCallback((resString: string): IResolution => {
    const [dimensions, refresh] = resString.split('@');
    const [width, height] = dimensions.split('x').map(Number);
    return {
      width,
      height,
      refreshRate: parseInt(refresh, 10)
    };
  }, []);

  // Memoized current resolution string
  const currentResolution = useMemo(() => 
    `${resolution.width}x${resolution.height}@${resolution.refreshRate}`,
    [resolution]
  );

  // Performance-tracked handlers
  const handleResolutionChange = useCallback(async (value: string) => {
    const start = performance.now();
    const newResolution = parseResolution(value);
    
    await onResolutionChange(newResolution);
    
    const latency = performance.now() - start;
    trackPerformance('resolution_change', latency);
    
    if (latency > UI_INTERACTIONS.INPUT_LATENCY_LIMIT) {
      console.warn(`Resolution change latency exceeded threshold: ${latency.toFixed(2)}ms`);
    }
  }, [onResolutionChange, parseResolution, trackPerformance]);

  const handleQualityChange = useCallback(async (value: string) => {
    const start = performance.now();
    
    await onQualityChange(value as RenderQuality);
    
    const latency = performance.now() - start;
    trackPerformance('quality_change', latency);
  }, [onQualityChange, trackPerformance]);

  const handleLidarQualityChange = useCallback(async (value: number, latency: number) => {
    trackPerformance('lidar_quality_change', latency);
    await onLidarQualityChange(value);
  }, [onLidarQualityChange, trackPerformance]);

  return (
    <SettingsContainer powerMode={powerMode}>
      <SettingGroup isActive={true} powerMode={powerMode}>
        <Label>Resolution</Label>
        <Dropdown
          options={RESOLUTION_OPTIONS}
          value={currentResolution}
          onChange={handleResolutionChange}
          powerMode={powerMode}
          hdrEnabled={theme.colorDepth.hdrEnabled}
          disabled={isPowerConstrained}
          width={300}
        />
      </SettingGroup>

      <SettingGroup isActive={true} powerMode={powerMode}>
        <Label>Quality Mode</Label>
        <Dropdown
          options={QUALITY_OPTIONS}
          value={quality}
          onChange={handleQualityChange}
          powerMode={powerMode}
          hdrEnabled={theme.colorDepth.hdrEnabled}
          width={300}
        />
      </SettingGroup>

      <SettingGroup isActive={!isPowerConstrained} powerMode={powerMode}>
        <Label>LiDAR Quality</Label>
        <CustomSlider
          value={lidarQuality}
          onChange={handleLidarQualityChange}
          min={0}
          max={100}
          step={1}
          label="LiDAR Quality"
          powerMode={powerMode}
          disabled={isPowerConstrained}
          criticalSetting={true}
        />
      </SettingGroup>
    </SettingsContainer>
  );
});

GraphicsSettings.displayName = 'GraphicsSettings';

export type { GraphicsSettingsProps, IResolution, RenderQuality };
export default GraphicsSettings;