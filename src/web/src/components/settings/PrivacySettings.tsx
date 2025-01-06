import React, { useState, useEffect, useCallback } from 'react'; // ^18.0.0
import styled from '@emotion/styled'; // ^11.11.0
import { auditLog } from '@security/audit-log'; // ^2.0.0
import { assessPrivacyImpact } from '@security/privacy-impact'; // ^1.0.0

import { useAuth } from '../../hooks/useAuth';
import { PrivacySettingsType, UserPreferencesType } from '../../interfaces/user.interface';

// Styled components for enhanced privacy UI
const PrivacyContainer = styled.div`
  padding: 24px;
  border-radius: 8px;
  background: var(--bg-elevated);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const SettingGroup = styled.div`
  margin-bottom: 24px;
`;

const SettingLabel = styled.label`
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  font-weight: 500;
  color: var(--text-primary);
`;

const SettingDescription = styled.p`
  color: var(--text-secondary);
  font-size: 14px;
  margin: 4px 0 12px;
`;

const WarningText = styled.span`
  color: var(--color-warning);
  font-size: 14px;
  margin-left: 8px;
`;

interface PrivacySettingsProps {
  onSave: (preferences: UserPreferencesType) => Promise<void>;
  className?: string;
  requireHardwareAuth?: boolean;
  isChildAccount?: boolean;
  parentalControls?: {
    allowDataSharing: boolean;
    allowFriendRequests: boolean;
  };
}

interface PrivacyState {
  showOnlineStatus: boolean;
  sharePlayHistory: boolean;
  allowFriendRequests: boolean;
  shareEnvironmentData: boolean;
  allowDataCollection: boolean;
  dataRetentionPeriod: number;
  requireParentalConsent: boolean;
  hardwareAuthEnabled: boolean;
}

export const PrivacySettings: React.FC<PrivacySettingsProps> = ({
  onSave,
  className,
  requireHardwareAuth = true,
  isChildAccount = false,
  parentalControls
}) => {
  const { user, authState, verifyHardwareToken } = useAuth();
  const [privacyState, setPrivacyState] = useState<PrivacyState>({
    showOnlineStatus: false,
    sharePlayHistory: false,
    allowFriendRequests: false,
    shareEnvironmentData: false,
    allowDataCollection: false,
    dataRetentionPeriod: 30,
    requireParentalConsent: isChildAccount,
    hardwareAuthEnabled: true
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial privacy settings
  useEffect(() => {
    if (user?.preferences?.privacy) {
      setPrivacyState(prevState => ({
        ...prevState,
        ...user.preferences.privacy
      }));
    }
  }, [user]);

  // Handle privacy setting changes with impact assessment
  const handlePrivacyChange = useCallback(async (
    settingKey: keyof PrivacyState,
    value: boolean | number
  ) => {
    try {
      // Check parental controls for child accounts
      if (isChildAccount && parentalControls) {
        if (settingKey === 'allowFriendRequests' && !parentalControls.allowFriendRequests) {
          throw new Error('Parental controls restrict friend requests');
        }
        if (settingKey === 'shareEnvironmentData' && !parentalControls.allowDataSharing) {
          throw new Error('Parental controls restrict data sharing');
        }
      }

      // Assess privacy impact of change
      const impact = await assessPrivacyImpact({
        setting: settingKey,
        newValue: value,
        userId: user?.id,
        isChildAccount
      });

      if (impact.riskLevel === 'high') {
        throw new Error(`High privacy risk: ${impact.details}`);
      }

      setPrivacyState(prev => ({
        ...prev,
        [settingKey]: value
      }));

      // Log privacy setting change
      await auditLog('privacy_setting_changed', {
        userId: user?.id,
        setting: settingKey,
        newValue: value,
        impact: impact.riskLevel
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update privacy setting');
    }
  }, [user, isChildAccount, parentalControls]);

  // Save privacy settings with hardware validation
  const handleSave = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsProcessing(true);
    setError(null);

    try {
      // Verify hardware token if required
      if (requireHardwareAuth) {
        const isValid = await verifyHardwareToken(authState?.hardwareToken || '');
        if (!isValid) {
          throw new Error('Hardware authentication failed');
        }
      }

      // Create updated preferences object
      const updatedPreferences: UserPreferencesType = {
        ...user?.preferences,
        privacy: {
          shareLocation: privacyState.showOnlineStatus,
          shareScanData: privacyState.shareEnvironmentData,
          dataRetentionDays: privacyState.dataRetentionPeriod,
          gdprConsent: privacyState.allowDataCollection
        }
      };

      // Save preferences
      await onSave(updatedPreferences);

      // Log successful update
      await auditLog('privacy_settings_updated', {
        userId: user?.id,
        settings: privacyState,
        hardwareAuthUsed: requireHardwareAuth
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save privacy settings');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <PrivacyContainer className={className}>
      <form onSubmit={handleSave}>
        <SettingGroup>
          <SettingLabel>
            <input
              type="checkbox"
              checked={privacyState.showOnlineStatus}
              onChange={e => handlePrivacyChange('showOnlineStatus', e.target.checked)}
              disabled={isProcessing}
            />
            Show Online Status
          </SettingLabel>
          <SettingDescription>
            Allow other players to see when you're online
          </SettingDescription>
        </SettingGroup>

        <SettingGroup>
          <SettingLabel>
            <input
              type="checkbox"
              checked={privacyState.sharePlayHistory}
              onChange={e => handlePrivacyChange('sharePlayHistory', e.target.checked)}
              disabled={isProcessing}
            />
            Share Play History
          </SettingLabel>
          <SettingDescription>
            Share your gaming activity with friends
            {isChildAccount && <WarningText>Requires parental approval</WarningText>}
          </SettingDescription>
        </SettingGroup>

        <SettingGroup>
          <SettingLabel>
            <input
              type="checkbox"
              checked={privacyState.allowFriendRequests}
              onChange={e => handlePrivacyChange('allowFriendRequests', e.target.checked)}
              disabled={isProcessing || (isChildAccount && !parentalControls?.allowFriendRequests)}
            />
            Allow Friend Requests
          </SettingLabel>
          <SettingDescription>
            Receive friend requests from other players
          </SettingDescription>
        </SettingGroup>

        <SettingGroup>
          <SettingLabel>
            <input
              type="checkbox"
              checked={privacyState.shareEnvironmentData}
              onChange={e => handlePrivacyChange('shareEnvironmentData', e.target.checked)}
              disabled={isProcessing || (isChildAccount && !parentalControls?.allowDataSharing)}
            />
            Share Environment Data
          </SettingLabel>
          <SettingDescription>
            Share LiDAR scan data with fleet members
          </SettingDescription>
        </SettingGroup>

        <SettingGroup>
          <SettingLabel>
            <input
              type="checkbox"
              checked={privacyState.allowDataCollection}
              onChange={e => handlePrivacyChange('allowDataCollection', e.target.checked)}
              disabled={isProcessing}
            />
            Allow Data Collection
          </SettingLabel>
          <SettingDescription>
            Help improve TALD UNIA by sharing anonymous usage data
          </SettingDescription>
        </SettingGroup>

        <SettingGroup>
          <SettingLabel>
            Data Retention Period (days)
          </SettingLabel>
          <input
            type="range"
            min={7}
            max={90}
            value={privacyState.dataRetentionPeriod}
            onChange={e => handlePrivacyChange('dataRetentionPeriod', parseInt(e.target.value))}
            disabled={isProcessing}
          />
          <SettingDescription>
            {privacyState.dataRetentionPeriod} days
          </SettingDescription>
        </SettingGroup>

        {error && (
          <SettingGroup>
            <WarningText>{error}</WarningText>
          </SettingGroup>
        )}

        <button
          type="submit"
          disabled={isProcessing}
        >
          {isProcessing ? 'Saving...' : 'Save Privacy Settings'}
        </button>
      </form>
    </PrivacyContainer>
  );
};

export default PrivacySettings;