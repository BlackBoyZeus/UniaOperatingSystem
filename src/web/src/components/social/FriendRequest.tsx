import React, { useCallback, useState } from 'react';
import styled from '@emotion/styled';
import debounce from 'lodash/debounce';
import { Button, ButtonProps } from '../common/Button';
import { IUser } from '../../interfaces/user.interface';
import { useNotification } from '@tald/notifications'; // v1.0.0

// Enhanced styled components with GPU acceleration and HDR support
const RequestContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: calc(var(--spacing-unit) * 2);
  background: var(--color-surface);
  border-radius: calc(var(--spacing-unit));
  margin-bottom: var(--spacing-unit);
  transform: var(--animation-gpu);
  will-change: transform;
  transition: transform var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1);
  contain: content;
  isolation: isolate;

  @media (dynamic-range: high) {
    box-shadow: var(--effect-glow);
  }
`;

const UserInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: calc(var(--spacing-unit) * 0.5);
  flex: 1;
`;

const Username = styled.span`
  font-family: var(--font-family-gaming);
  font-feature-settings: var(--font-feature-gaming);
  font-size: 1rem;
  color: var(--color-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
`;

const Distance = styled.span`
  font-size: 0.875rem;
  color: var(--color-secondary);
  display: flex;
  align-items: center;
  gap: calc(var(--spacing-unit) * 0.5);
`;

const Actions = styled.div`
  display: flex;
  gap: var(--spacing-unit);
  transform: var(--animation-gpu);
  will-change: transform;
`;

// Enhanced props interface with real-time features
interface FriendRequestProps {
  user: IUser;
  distance: number;
  lastUpdate: Date;
  onAccept: (userId: string) => Promise<void>;
  onReject: (userId: string) => Promise<void>;
  className?: string;
  isRateLimited?: boolean;
  retryCount?: number;
}

export const FriendRequest: React.FC<FriendRequestProps> = ({
  user,
  distance,
  lastUpdate,
  onAccept,
  onReject,
  className,
  isRateLimited = false,
  retryCount = 0,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const { showNotification } = useNotification();

  // Optimized accept handler with debouncing and error recovery
  const handleAccept = useCallback(
    debounce(async (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      
      if (isRateLimited || isProcessing) {
        showNotification({
          type: 'warning',
          message: 'Please wait before trying again',
          duration: 3000,
        });
        return;
      }

      try {
        setIsProcessing(true);
        await onAccept(user.id);
        showNotification({
          type: 'success',
          message: `Friend request accepted from ${user.username}`,
          duration: 3000,
        });
      } catch (error) {
        showNotification({
          type: 'error',
          message: 'Failed to accept friend request',
          duration: 5000,
        });
        
        if (retryCount < 3) {
          setTimeout(() => handleAccept(event), 1000 * (retryCount + 1));
        }
      } finally {
        setIsProcessing(false);
      }
    }, 300),
    [user, onAccept, isRateLimited, isProcessing, retryCount, showNotification]
  );

  // Secure reject handler with validation
  const handleReject = useCallback(
    debounce(async (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();

      if (isRateLimited || isProcessing) {
        showNotification({
          type: 'warning',
          message: 'Please wait before trying again',
          duration: 3000,
        });
        return;
      }

      try {
        setIsProcessing(true);
        await onReject(user.id);
        showNotification({
          type: 'info',
          message: `Friend request rejected from ${user.username}`,
          duration: 3000,
        });
      } catch (error) {
        showNotification({
          type: 'error',
          message: 'Failed to reject friend request',
          duration: 5000,
        });
      } finally {
        setIsProcessing(false);
      }
    }, 300),
    [user, onReject, isRateLimited, isProcessing, showNotification]
  );

  return (
    <RequestContainer className={className}>
      <UserInfo>
        <Username>{user.username}</Username>
        <Distance>
          {distance.toFixed(1)}m away • {user.status}
          {lastUpdate && ` • Last updated ${new Date(lastUpdate).toLocaleTimeString()}`}
        </Distance>
      </UserInfo>
      <Actions>
        <Button
          variant="primary"
          size="small"
          onClick={handleAccept}
          disabled={isProcessing || isRateLimited}
          enableHaptic
          powerSaveAware
          hdrMode="auto"
        >
          Accept
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={handleReject}
          disabled={isProcessing || isRateLimited}
          enableHaptic
          powerSaveAware
          hdrMode="auto"
        >
          Reject
        </Button>
      </Actions>
    </RequestContainer>
  );
};

export type { FriendRequestProps };