import { useCallback } from 'react'; // ^18.0.0
import { useAuthContext } from '../contexts/AuthContext';
import { IUser, IUserAuth, UserRoleType, ISecurityEvent } from '../interfaces/user.interface';

/**
 * Interface for the useAuth hook return value with enhanced security features
 */
interface IUseAuth {
  user: IUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  securityEvents: ISecurityEvent[];
  securityLevel: string;
  login: (username: string, password: string, hardwareToken: string) => Promise<void>;
  logout: () => Promise<void>;
  hasRole: (requiredRole: UserRoleType) => boolean;
  monitorSecurityEvents: () => Promise<void>;
}

/**
 * Custom hook for managing authentication state and operations with enhanced security features
 * Implements OAuth 2.0 + JWT based authentication with hardware-backed keys and TPM integration
 */
export const useAuth = (): IUseAuth => {
  const {
    user,
    authState,
    isLoading,
    error,
    securityEvents,
    login: contextLogin,
    logout: contextLogout,
    hasRole: contextHasRole,
    checkSecurityStatus
  } = useAuthContext();

  /**
   * Enhanced login handler with hardware token validation and security monitoring
   */
  const login = useCallback(
    async (username: string, password: string, hardwareToken: string): Promise<void> => {
      try {
        await contextLogin(username, password, hardwareToken);
      } catch (err) {
        // Error is already handled by context, just propagate
        throw err;
      }
    },
    [contextLogin]
  );

  /**
   * Enhanced logout handler with security cleanup
   */
  const logout = useCallback(async (): Promise<void> => {
    try {
      await contextLogout();
    } catch (err) {
      // Error is already handled by context, just propagate
      throw err;
    }
  }, [contextLogout]);

  /**
   * Role verification with real-time validation
   */
  const hasRole = useCallback(
    (requiredRole: UserRoleType): boolean => {
      return contextHasRole(requiredRole);
    },
    [contextHasRole]
  );

  /**
   * Security event monitoring with threat detection
   */
  const monitorSecurityEvents = useCallback(async (): Promise<void> => {
    try {
      await checkSecurityStatus();
    } catch (err) {
      // Error is already handled by context, just propagate
      throw err;
    }
  }, [checkSecurityStatus]);

  /**
   * Computed authentication status
   */
  const isAuthenticated = Boolean(user && authState?.accessToken);

  /**
   * Current security level based on user and hardware capabilities
   */
  const securityLevel = user?.securityLevel || 'NONE';

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    securityEvents,
    securityLevel,
    login,
    logout,
    hasRole,
    monitorSecurityEvents
  };
};

export default useAuth;