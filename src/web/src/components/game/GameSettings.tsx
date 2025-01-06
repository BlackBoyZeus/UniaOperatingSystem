import React, { useCallback, useEffect, useState, useMemo } from 'react';
import styled from 'styled-components';
import { debounce } from 'lodash'; // ^4.17.21
import CustomSlider from '../common/Slider';
import Dropdown from '../common/Dropdown';
import { GameService } from '../../services/game.service';
import { RenderQuality, IWebRenderState } from '../../interfaces/game.interface';
import { UI_CONSTANTS } from '../../constants/ui.constants';

/**
 * Styled components with GPU acceleration and power-aware animations
 */
const SettingsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  transform: translateZ(0);
  will-change: transform, opacity;
  transition: all ${UI_CONSTANTS.ANIMATION_TIMINGS.SETTINGS}ms ease-in-out;
`;

const SettingRow = styled.div<{ disabled?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  opacity: ${props => props.disabled ? 0.5 : 1};
  transition: opacity ${UI_CONSTANTS.ANIMATION_TIMINGS.SETTINGS}ms ease;
`;

const SettingLabel = styled.label`
  font-family: ${props => props.theme.typography.gaming.family};
  font-weight: ${props => props.theme.typography.gaming.weights.regular};
  color: ${props => props.theme.colorDepth.hdrEnabled ? 
    'color(display-p3 1 1 1)' : 
    props.theme.palette.text.primary};
`;

const PerformanceIndicator = styled.div<{ performance: number }>`
  font-size: 12px;
  color: ${props => {
    if (props.performance >= 58) return 'color(display-p3 0 1 0)';
    if (props.performance >= 30) return 'color(display-p3 1 1 0)';
    return 'color(display-p3 1 0 0)';
  }};
`;

/**
 * Interface for GameSettings component props
 */
interface GameSettingsProps {
  initialSettings: IWebRenderState;
  onSettingsChange: (settings: IWebRenderState) => void;
  disabled?: boolean;
  onPerformanceData?: (data: { fps: number; latency: number }) => void;
}

/**
 * Resolution options for the game
 */
const RESOLUTION_OPTIONS = [
  { value: '1920x1080', label: '1080p' },
  { value: '2560x1440', label: '1440p' },
  { value: '3840x2160', label: '4K' }
] as const;

/**
 * Quality options mapping
 */
const QUALITY_OPTIONS = Object.values(RenderQuality).map(quality => ({
  value: quality,
  label: quality.charAt(0) + quality.slice(1).toLowerCase()
}));

/**
 * GameSettings component for managing game graphics and performance settings
 */
const GameSettings: React.FC<GameSettingsProps> = React.memo(({
  initialSettings,
  onSettingsChange,
  disabled = false,
  onPerformanceData
}) => {
  // State management
  const [settings, setSettings] = useState<IWebRenderState>(initialSettings);
  const [performance, setPerformance] = useState({ fps: 60, latency: 0 });

  // Memoized game service instance
  const gameService = useMemo(() => new GameService(), []);

  /**
   * Debounced settings update handler
   */
  const debouncedSettingsChange = useMemo(
    () => debounce((newSettings: IWebRenderState) => {
      onSettingsChange(newSettings);
    }, 150),
    [onSettingsChange]
  );

  /**
   * Handle quality setting changes with performance monitoring
   */
  const handleQualityChange = useCallback((quality: string) => {
    if (disabled) return;

    const newSettings = {
      ...settings,
      quality: quality as RenderQuality
    };

    setSettings(newSettings);
    debouncedSettingsChange(newSettings);

    // Monitor performance impact
    gameService.monitorPerformance().then(metrics => {
      setPerformance(metrics);
      onPerformanceData?.(metrics);
    });
  }, [settings, disabled, debouncedSettingsChange, gameService, onPerformanceData]);

  /**
   * Handle resolution changes with power-aware adjustments
   */
  const handleResolutionChange = useCallback((resolution: string) => {
    if (disabled) return;

    const [width, height] = resolution.split('x').map(Number);
    const newSettings = {
      ...settings,
      resolution: { width, height }
    };

    setSettings(newSettings);
    debouncedSettingsChange(newSettings);

    // Update environment for new resolution
    gameService.updateEnvironment(newSettings).then(metrics => {
      setPerformance(metrics);
      onPerformanceData?.(metrics);
    });
  }, [settings, disabled, debouncedSettingsChange, gameService, onPerformanceData]);

  /**
   * Handle LiDAR overlay toggle
   */
  const handleLidarOverlayChange = useCallback((enabled: boolean) => {
    if (disabled) return;

    const newSettings = {
      ...settings,
      lidarOverlayEnabled: enabled
    };

    setSettings(newSettings);
    debouncedSettingsChange(newSettings);
  }, [settings, disabled, debouncedSettingsChange]);

  /**
   * Monitor performance on mount and settings changes
   */
  useEffect(() => {
    if (disabled) return;

    const performanceMonitor = setInterval(() => {
      gameService.monitorPerformance().then(metrics => {
        setPerformance(metrics);
        onPerformanceData?.(metrics);
      });
    }, 1000);

    return () => clearInterval(performanceMonitor);
  }, [disabled, gameService, onPerformanceData]);

  return (
    <SettingsContainer>
      <SettingRow disabled={disabled}>
        <SettingLabel>Quality</SettingLabel>
        <Dropdown
          options={QUALITY_OPTIONS}
          value={settings.quality}
          onChange={handleQualityChange}
          disabled={disabled}
          width={150}
        />
      </SettingRow>

      <SettingRow disabled={disabled}>
        <SettingLabel>Resolution</SettingLabel>
        <Dropdown
          options={RESOLUTION_OPTIONS}
          value={`${settings.resolution.width}x${settings.resolution.height}`}
          onChange={handleResolutionChange}
          disabled={disabled}
          width={150}
        />
      </SettingRow>

      <SettingRow disabled={disabled}>
        <SettingLabel>LiDAR Overlay</SettingLabel>
        <CustomSlider
          value={settings.lidarOverlayEnabled ? 1 : 0}
          onChange={value => handleLidarOverlayChange(Boolean(value))}
          min={0}
          max={1}
          step={1}
          disabled={disabled}
          label="LiDAR Overlay"
        />
      </SettingRow>

      <PerformanceIndicator performance={performance.fps}>
        FPS: {performance.fps} | Latency: {performance.latency.toFixed(1)}ms
      </PerformanceIndicator>
    </SettingsContainer>
  );
});

GameSettings.displayName = 'GameSettings';

export type { GameSettingsProps };
export default GameSettings;