import React, { useState, useCallback, useEffect } from 'react';
import styled from '@emotion/styled';
import { useGPUAcceleration } from '@gaming-ui/acceleration';
import { useSecurityMonitor } from '@gaming-ui/security';

import Button from '../common/Button';
import Input from '../common/Input';
import { useAuth } from '../../hooks/useAuth';
import type { IUserProfile } from '../../interfaces/user.interface';

// Enhanced props interface with security features
interface ProfileEditProps {
  onSave: (profile: IUserProfile, hardwareToken: string) => Promise<void>;
  onCancel: () => void;
  initialProfile: IUserProfile;
  securityLevel: string;
}

// GPU-accelerated styled components
const FormContainer = styled.form<{ isGPUEnabled: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1.5rem;
  background: color(display-p3 var(--color-surface));
  border-radius: 8px;
  box-shadow: var(--effect-glow);
  max-width: 480px;
  width: 100%;
  color-space: display-p3;
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  will-change: ${props => props.isGPUEnabled ? 'transform, opacity' : 'auto'};
  transform: translateZ(0);
  backface-visibility: hidden;
  content-visibility: auto;
  contain: content;

  @media (dynamic-range: high) {
    background: color(display-p3 var(--color-surface-hdr));
    box-shadow: var(--effect-glow-hdr);
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
  justify-content: flex-end;
  transform: translateZ(0);
  backface-visibility: hidden;
`;

const ErrorMessage = styled.div`
  color: color(display-p3 1 0 0);
  font-size: 0.875rem;
  margin-top: 0.5rem;
  opacity: 0.9;
`;

const ProfileEdit: React.FC<ProfileEditProps> = ({
  onSave,
  onCancel,
  initialProfile,
  securityLevel
}) => {
  // State management with security validation
  const [profile, setProfile] = useState<IUserProfile>(initialProfile);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Custom hooks for enhanced features
  const { user, validateHardwareToken } = useAuth();
  const { enableGPU, isGPUEnabled } = useGPUAcceleration();
  const { monitorSecurityEvents } = useSecurityMonitor();

  // Enable GPU acceleration on mount
  useEffect(() => {
    enableGPU();
  }, [enableGPU]);

  // Security monitoring setup
  useEffect(() => {
    const securityCheck = async () => {
      const events = await monitorSecurityEvents();
      if (events.some(event => event.severity === 'critical')) {
        setErrors(prev => ({
          ...prev,
          security: 'Critical security event detected'
        }));
      }
    };

    const intervalId = setInterval(securityCheck, 30000);
    return () => clearInterval(intervalId);
  }, [monitorSecurityEvents]);

  // Enhanced input change handler with validation
  const handleInputChange = useCallback((field: keyof IUserProfile) => (
    value: string
  ) => {
    setProfile(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear field-specific error
    setErrors(prev => ({
      ...prev,
      [field]: undefined
    }));
  }, []);

  // Enhanced form submission with security validation
  const handleSubmit = useCallback(async (
    event: React.FormEvent
  ) => {
    event.preventDefault();
    setIsSubmitting(true);
    setErrors({});

    try {
      // Validate hardware token
      const hardwareToken = await validateHardwareToken();
      if (!hardwareToken) {
        throw new Error('Hardware token validation failed');
      }

      // Validate security level
      if (securityLevel !== user?.securityLevel) {
        throw new Error('Security level mismatch');
      }

      // Validate profile data
      if (!profile.displayName?.trim()) {
        throw new Error('Display name is required');
      }

      // Save profile with hardware token
      await onSave(profile, hardwareToken);

    } catch (error) {
      setErrors({
        submit: error instanceof Error ? error.message : 'Profile update failed'
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [profile, securityLevel, user, validateHardwareToken, onSave]);

  return (
    <FormContainer 
      onSubmit={handleSubmit}
      isGPUEnabled={isGPUEnabled}
      data-security-level={securityLevel}
    >
      <Input
        id="displayName"
        name="displayName"
        type="text"
        value={profile.displayName}
        onChange={handleInputChange('displayName')}
        placeholder="Display Name"
        error={errors.displayName}
        required
        validation
        securityLevel={securityLevel}
        gpuAccelerated={isGPUEnabled}
      />

      <Input
        id="avatar"
        name="avatar"
        type="text"
        value={profile.avatar}
        onChange={handleInputChange('avatar')}
        placeholder="Avatar URL"
        error={errors.avatar}
        validation
        securityLevel={securityLevel}
        gpuAccelerated={isGPUEnabled}
      />

      {Object.entries(profile.preferences).map(([key, value]) => (
        <Input
          key={key}
          id={key}
          name={key}
          type={typeof value === 'boolean' ? 'checkbox' : 'text'}
          value={String(value)}
          onChange={handleInputChange(`preferences.${key}` as keyof IUserProfile)}
          placeholder={key.charAt(0).toUpperCase() + key.slice(1)}
          error={errors[key]}
          validation
          securityLevel={securityLevel}
          gpuAccelerated={isGPUEnabled}
        />
      ))}

      {errors.submit && (
        <ErrorMessage role="alert">
          {errors.submit}
        </ErrorMessage>
      )}

      <ButtonGroup>
        <Button
          type="button"
          variant="secondary"
          onClick={onCancel}
          disabled={isSubmitting}
          enableHaptic
          powerSaveAware
        >
          Cancel
        </Button>
        <Button
          type="submit"
          variant="primary"
          disabled={isSubmitting}
          enableHaptic
          powerSaveAware
        >
          {isSubmitting ? 'Saving...' : 'Save Profile'}
        </Button>
      </ButtonGroup>
    </FormContainer>
  );
};

export default ProfileEdit;
export type { ProfileEditProps };