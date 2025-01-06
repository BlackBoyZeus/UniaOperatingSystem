import React, { useCallback, useEffect, useState } from 'react';
import styled from '@emotion/styled';
import { useNavigate } from 'react-router-dom';

import DashboardLayout from '../layouts/DashboardLayout';
import useAuth from '../hooks/useAuth';
import Button from '../components/common/Button';
import Icon from '../components/common/Icon';
import Dropdown from '../components/common/Dropdown';

// Enhanced state interface with security monitoring
interface ProfilePageState {
  isEditing: boolean;
  error: string | null;
  securityStatus: {
    hardwareTokenValid: boolean;
    lastValidation: number;
    securityLevel: string;
  };
  hardwareValidation: {
    inProgress: boolean;
    lastCheck: number;
    status: 'valid' | 'invalid' | 'pending';
  };
}

// GPU-accelerated container with HDR support
const SecureProfileContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;

  /* GPU acceleration optimizations */
  will-change: transform;
  transform: translateZ(0);
  backface-visibility: hidden;
  
  /* HDR color space support */
  color-space: display-p3;
  
  /* Power-aware animations */
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
`;

const ProfileHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  background: color(display-p3 0.15 0.15 0.2);
  border-radius: 8px;
  box-shadow: 0 2px 8px color(display-p3 0 0 0 / 0.2);
`;

const SecurityAlert = styled.div<{ severity: 'high' | 'medium' | 'low' }>`
  color: ${props => 
    props.severity === 'high' 
      ? 'color(display-p3 1 0 0)' 
      : props.severity === 'medium'
        ? 'color(display-p3 1 0.5 0)'
        : 'color(display-p3 0 0.8 0)'};
  padding: 12px;
  border-radius: 8px;
  background-color: color(display-p3 1 0 0 / 0.1);
  margin-bottom: 16px;
`;

const ProfileSection = styled.section`
  background: color(display-p3 0.2 0.2 0.25);
  padding: 24px;
  border-radius: 8px;
  transform: translateZ(0);
  will-change: transform, opacity;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
`;

const ProfilePage: React.FC = React.memo(() => {
  const navigate = useNavigate();
  const { user, updateProfile, validateUserRole } = useAuth();
  
  const [state, setState] = useState<ProfilePageState>({
    isEditing: false,
    error: null,
    securityStatus: {
      hardwareTokenValid: false,
      lastValidation: 0,
      securityLevel: 'standard'
    },
    hardwareValidation: {
      inProgress: false,
      lastCheck: 0,
      status: 'pending'
    }
  });

  // Security validation effect
  useEffect(() => {
    const validateSecurity = async () => {
      try {
        setState(prev => ({
          ...prev,
          hardwareValidation: { ...prev.hardwareValidation, inProgress: true }
        }));

        const isValid = await validateUserRole(user?.role || 'USER');
        
        setState(prev => ({
          ...prev,
          securityStatus: {
            hardwareTokenValid: isValid,
            lastValidation: Date.now(),
            securityLevel: user?.securityLevel || 'standard'
          },
          hardwareValidation: {
            inProgress: false,
            lastCheck: Date.now(),
            status: isValid ? 'valid' : 'invalid'
          }
        }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Security validation failed',
          hardwareValidation: {
            ...prev.hardwareValidation,
            inProgress: false,
            status: 'invalid'
          }
        }));
      }
    };

    validateSecurity();
  }, [user, validateUserRole]);

  // Enhanced profile update with security validation
  const handleProfileUpdate = useCallback(async (updatedData: any) => {
    try {
      setState(prev => ({ ...prev, isEditing: false, error: null }));
      
      // Validate hardware token before update
      if (!state.securityStatus.hardwareTokenValid) {
        throw new Error('Invalid hardware token');
      }

      await updateProfile(updatedData);
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: 'Profile update failed: Invalid security context'
      }));
    }
  }, [updateProfile, state.securityStatus.hardwareTokenValid]);

  if (!user) {
    navigate('/login');
    return null;
  }

  return (
    <DashboardLayout>
      <SecureProfileContainer>
        {state.error && (
          <SecurityAlert severity="high">
            <Icon name="warning" size={20} powerMode="BALANCED" />
            {state.error}
          </SecurityAlert>
        )}

        <ProfileHeader>
          <h1>Profile Settings</h1>
          <div>
            <Icon 
              name={state.securityStatus.hardwareTokenValid ? 'security' : 'security-warning'} 
              size={24}
              animate={!state.securityStatus.hardwareTokenValid}
              powerMode="BALANCED"
            />
            Security Level: {state.securityStatus.securityLevel}
          </div>
        </ProfileHeader>

        <ProfileSection>
          <h2>User Information</h2>
          <div>
            <strong>Username:</strong> {user.username}
            <br />
            <strong>Email:</strong> {user.email}
            <br />
            <strong>Role:</strong> {user.role}
          </div>
        </ProfileSection>

        <ProfileSection>
          <h2>Device Capabilities</h2>
          <div>
            <strong>LiDAR Support:</strong> {user.deviceCapabilities.lidarSupported ? 'Yes' : 'No'}
            <br />
            <strong>Mesh Network:</strong> {user.deviceCapabilities.meshNetworkSupported ? 'Yes' : 'No'}
            <br />
            <strong>Vulkan Version:</strong> {user.deviceCapabilities.vulkanVersion}
            <br />
            <strong>Security Level:</strong> {user.deviceCapabilities.hardwareSecurityLevel}
          </div>
        </ProfileSection>

        <ProfileSection>
          <h2>Security Settings</h2>
          <Button
            variant="primary"
            size="medium"
            enableHaptic
            hdrMode="auto"
            powerSaveAware
            onClick={() => setState(prev => ({ ...prev, isEditing: true }))}
          >
            Update Security Settings
          </Button>
        </ProfileSection>
      </SecureProfileContainer>
    </DashboardLayout>
  );
});

ProfilePage.displayName = 'ProfilePage';

export default ProfilePage;