import React, { useMemo } from 'react';
import styled from '@emotion/styled';
import { ErrorBoundary } from 'react-error-boundary'; // ^4.0.0
import { ILidarScanState } from '../../interfaces/lidar.interface';
import { useLidar } from '../../hooks/useLidar';
import Progress from '../common/Progress';
import { LIDAR_SCAN_SETTINGS } from '../../constants/lidar.constants';
import { THEME_SETTINGS, ANIMATION_TIMINGS } from '../../constants/ui.constants';

// Types
type PowerMode = 'performance' | 'balanced' | 'powersave';
type ColorMode = 'HDR' | 'SDR';

interface LidarStatsProps {
  className?: string;
  powerMode?: PowerMode;
  colorMode?: ColorMode;
}

// Styled components with GPU acceleration
const StatsContainer = styled.div<{ powerMode: PowerMode }>`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  padding: 16px;
  background: ${THEME_SETTINGS.DARK.background};
  border-radius: 8px;
  transform: translateZ(0);
  will-change: transform;
  transition: all ${props => 
    ANIMATION_TIMINGS.POWER_MODES[props.powerMode.toUpperCase()].transitionMultiplier * 
    ANIMATION_TIMINGS.TRANSITION_DURATION}ms ${ANIMATION_TIMINGS.EASING.DEFAULT};
`;

const StatItem = styled.div<{
  quality: 'HIGH' | 'MEDIUM' | 'LOW';
  isError: boolean;
  colorMode: ColorMode;
}>`
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  background: ${props => props.isError ? 
    'rgba(255, 0, 0, 0.1)' : 
    'rgba(255, 255, 255, 0.05)'
  };
  border-radius: 4px;
  color: ${props => props.colorMode === 'HDR' ? 
    THEME_SETTINGS.DARK.hdrColors.primary :
    THEME_SETTINGS.DARK.primary
  };
  transform: translateZ(0);
  will-change: transform, opacity;
`;

const StatLabel = styled.span`
  font-size: 14px;
  color: ${THEME_SETTINGS.DARK.contrast.medium};
`;

const StatValue = styled.span`
  font-size: 18px;
  font-weight: 500;
  color: ${THEME_SETTINGS.DARK.contrast.high};
`;

const ErrorContainer = styled.div`
  padding: 16px;
  background: rgba(255, 0, 0, 0.1);
  border-radius: 8px;
  color: ${THEME_SETTINGS.DARK.contrast.high};
`;

// Memoized helper functions
const calculateQualityPercentage = (metadata: ILidarScanState['metadata']) => {
  if (!metadata) return 0;
  
  const scannerHealth = metadata.scannerHealth || 0;
  const environmentalFactors = metadata.environmentalFactors || {};
  const ambientScore = 1 - (environmentalFactors.ambientLight || 0);
  
  return Math.round((scannerHealth + ambientScore) * 50);
};

// Main component
const LidarStats: React.FC<LidarStatsProps> = React.memo(({ 
  className,
  powerMode = 'balanced',
  colorMode = 'HDR'
}) => {
  const { scanState, performanceMetrics } = useLidar();
  
  // Memoized calculations
  const qualityPercentage = useMemo(() => 
    calculateQualityPercentage(scanState.metadata),
    [scanState.metadata]
  );

  const scanFrequency = useMemo(() => 
    scanState.isActive ? LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY : 0,
    [scanState.isActive]
  );

  const processingLatency = useMemo(() => 
    performanceMetrics?.processingLatency || 0,
    [performanceMetrics]
  );

  if (scanState.error) {
    return (
      <ErrorContainer>
        Error: {scanState.error.message}
      </ErrorContainer>
    );
  }

  return (
    <StatsContainer className={className} powerMode={powerMode}>
      <StatItem 
        quality={performanceMetrics?.quality || 'MEDIUM'}
        isError={false}
        colorMode={colorMode}
      >
        <StatLabel>Scan Frequency</StatLabel>
        <StatValue>{scanFrequency} Hz</StatValue>
        <Progress 
          value={(scanFrequency / LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY) * 100}
          powerMode={powerMode}
          color="primary"
          size="small"
        />
      </StatItem>

      <StatItem 
        quality={performanceMetrics?.quality || 'MEDIUM'}
        isError={false}
        colorMode={colorMode}
      >
        <StatLabel>Resolution</StatLabel>
        <StatValue>{LIDAR_SCAN_SETTINGS.RESOLUTION} cm</StatValue>
        <Progress 
          value={100}
          powerMode={powerMode}
          color="secondary"
          size="small"
        />
      </StatItem>

      <StatItem 
        quality={performanceMetrics?.quality || 'MEDIUM'}
        isError={processingLatency > LIDAR_SCAN_SETTINGS.PROCESSING_LATENCY_LIMIT}
        colorMode={colorMode}
      >
        <StatLabel>Processing Latency</StatLabel>
        <StatValue>{processingLatency.toFixed(1)} ms</StatValue>
        <Progress 
          value={(processingLatency / LIDAR_SCAN_SETTINGS.PROCESSING_LATENCY_LIMIT) * 100}
          powerMode={powerMode}
          color={processingLatency > LIDAR_SCAN_SETTINGS.PROCESSING_LATENCY_LIMIT ? 'accent' : 'primary'}
          size="small"
        />
      </StatItem>

      <StatItem 
        quality={performanceMetrics?.quality || 'MEDIUM'}
        isError={false}
        colorMode={colorMode}
      >
        <StatLabel>Scan Quality</StatLabel>
        <StatValue>{qualityPercentage}%</StatValue>
        <Progress 
          value={qualityPercentage}
          powerMode={powerMode}
          color={qualityPercentage > 80 ? 'primary' : 'accent'}
          size="small"
        />
      </StatItem>
    </StatsContainer>
  );
});

// Error boundary wrapper
const LidarStatsWithErrorBoundary: React.FC<LidarStatsProps> = (props) => (
  <ErrorBoundary
    fallback={<ErrorContainer>Failed to load LiDAR statistics</ErrorContainer>}
    onError={(error) => console.error('LidarStats Error:', error)}
  >
    <LidarStats {...props} />
  </ErrorBoundary>
);

LidarStatsWithErrorBoundary.displayName = 'LidarStats';

export default LidarStatsWithErrorBoundary;