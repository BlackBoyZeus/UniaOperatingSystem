import React from 'react'; // ^18.0.0
import styled from '@emotion/styled'; // ^11.11.0
import { Navigate, useLocation } from 'react-router-dom'; // ^6.0.0

import { useAuthContext } from '../contexts/AuthContext';
import Loading from '../components/common/Loading';
import { THEME_SETTINGS, Z_INDEX, ANIMATION_TIMINGS } from '../constants/ui.constants';

// Power mode type for performance optimization
type PowerMode = 'high' | 'balanced' | 'low';

// Props interface with HDR and power-aware features
interface AuthLayoutProps {
  children: React.ReactNode;
  className?: string;
  enableHDR?: boolean;
  powerMode?: PowerMode;
}

// GPU-accelerated container with HDR support
const StyledAuthContainer = styled.div<{
  hasHDR: boolean;
  powerMode: PowerMode;
}>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: ${({ hasHDR }) => 
    hasHDR ? 'color(display-p3 0.07 0.07 0.07)' : THEME_SETTINGS.DARK.background};
  padding: 2rem;
  will-change: transform;
  transform: translateZ(0);
  color-space: ${({ hasHDR }) => hasHDR ? 'display-p3' : 'srgb'};
  transition: background-color ${({ powerMode }) => 
    ANIMATION_TIMINGS.POWER_MODES[powerMode === 'high' 
      ? 'HIGH_PERFORMANCE' 
      : powerMode === 'low' 
        ? 'POWER_SAVER' 
        : 'BALANCED'].transitionMultiplier * 300}ms 
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
`;

// Power-aware content wrapper with gaming aesthetics
const StyledAuthContent = styled.div<{ powerMode: PowerMode }>`
  width: 100%;
  max-width: 400px;
  background: rgba(30, 30, 30, 0.95);
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 0 20px rgba(0, 255, 0, 0.1);
  animation: fadeIn ${({ powerMode }) => 
    ANIMATION_TIMINGS.POWER_MODES[powerMode === 'high' 
      ? 'HIGH_PERFORMANCE' 
      : powerMode === 'low' 
        ? 'POWER_SAVER' 
        : 'BALANCED'].transitionMultiplier * 300}ms 
    ${ANIMATION_TIMINGS.EASING.DEFAULT};
  will-change: transform, opacity;
  transform: translateZ(0);
  
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translate3d(0, -20px, 0);
    }
    to {
      opacity: 1;
      transform: translate3d(0, 0, 0);
    }
  }
`;

// Gaming-themed error message with animations
const StyledErrorMessage = styled.div`
  color: ${THEME_SETTINGS.DARK.accent};
  text-align: center;
  margin-bottom: 1rem;
  font-family: gaming-font, sans-serif;
  animation: shake 0.5s ${ANIMATION_TIMINGS.EASING.SHARP};
  will-change: transform;
  transform: translateZ(0);

  @keyframes shake {
    0%, 100% { transform: translate3d(0, 0, 0); }
    10%, 30%, 50%, 70%, 90% { transform: translate3d(-5px, 0, 0); }
    20%, 40%, 60%, 80% { transform: translate3d(5px, 0, 0); }
  }
`;

/**
 * Authentication layout component with gaming optimizations
 */
const AuthLayout: React.FC<AuthLayoutProps> = React.memo(({
  children,
  className,
  enableHDR = false,
  powerMode = 'balanced'
}) => {
  const { isLoading, error } = useAuthContext();
  const location = useLocation();

  // Protected route handling
  if (location.pathname !== '/login' && !isLoading && !error) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return (
    <StyledAuthContainer 
      className={className}
      hasHDR={enableHDR}
      powerMode={powerMode}
    >
      <StyledAuthContent powerMode={powerMode}>
        {isLoading ? (
          <Loading 
            size="medium"
            color="primary"
            powerMode={powerMode}
            hdrEnabled={enableHDR}
            text="Authenticating..."
          />
        ) : (
          <>
            {error && (
              <StyledErrorMessage role="alert">
                {error}
              </StyledErrorMessage>
            )}
            {children}
          </>
        )}
      </StyledAuthContent>
    </StyledAuthContainer>
  );
});

AuthLayout.displayName = 'AuthLayout';

export default AuthLayout;