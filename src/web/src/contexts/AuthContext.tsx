import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'; // ^18.0.0

import AuthService from '../services/auth.service';
import { IUser, IUserAuth, UserRoleType, ISecurityEvent } from '../interfaces/user.interface';

// Error message for context usage outside provider
const AUTH_CONTEXT_ERROR = 'useAuth must be used within an AuthProvider';

// Security monitoring intervals
const TOKEN_REFRESH_INTERVAL = 900000; // 15 minutes
const MAX_AUTH_ATTEMPTS = 5;
const SECURITY_CHECK_INTERVAL = 30000; // 30 seconds

/**
 * Enhanced authentication context interface with security monitoring
 */
interface IAuthContext {
  user: IUser | null;
  authState: IUserAuth | null;
  isLoading: boolean;
  error: string | null;
  securityEvents: ISecurityEvent[];
  login: (username: string, password: string, hardwareToken: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => boolean;
  hasRole: (requiredRole: UserRoleType) => boolean;
  checkSecurityStatus: () => Promise<ISecurityEvent[]>;
}

// Create context with security monitoring
const AuthContext = createContext<IAuthContext | null>(null);

/**
 * Enhanced hook to access authentication context with security monitoring
 */
export const useAuthContext = (): IAuthContext => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error(AUTH_CONTEXT_ERROR);
  }
  return context;
};

/**
 * Enhanced authentication provider with security monitoring
 */
export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<IUser | null>(null);
  const [authState, setAuthState] = useState<IUserAuth | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [securityEvents, setSecurityEvents] = useState<ISecurityEvent[]>([]);
  const [authAttempts, setAuthAttempts] = useState<number>(0);

  /**
   * Enhanced login process with security monitoring
   */
  const handleLogin = useCallback(async (
    username: string,
    password: string,
    hardwareToken: string
  ): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);

      // Check authentication attempts
      if (authAttempts >= MAX_AUTH_ATTEMPTS) {
        throw new Error('Maximum authentication attempts exceeded');
      }

      // Validate hardware token
      const isValidToken = await AuthService.validateHardwareToken(hardwareToken);
      if (!isValidToken) {
        setAuthAttempts(prev => prev + 1);
        throw new Error('Invalid hardware token');
      }

      // Perform login
      const authData = await AuthService.login(username, password, hardwareToken);
      const userState = await AuthService.getAuthState();

      setUser(userState);
      setAuthState(authData);
      setAuthAttempts(0);

      // Start security monitoring
      startSecurityMonitoring();

      // Log successful login
      AuthService.logAuditEvent('login_success', {
        userId: userState.id,
        timestamp: new Date().toISOString()
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Authentication failed';
      setError(errorMessage);
      
      // Log failed login attempt
      AuthService.logAuditEvent('login_failure', {
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [authAttempts]);

  /**
   * Enhanced logout process with security cleanup
   */
  const handleLogout = useCallback(async (): Promise<void> => {
    try {
      await AuthService.logout();
      
      // Log successful logout
      AuthService.logAuditEvent('logout_success', {
        userId: user?.id,
        timestamp: new Date().toISOString()
      });

    } finally {
      setUser(null);
      setAuthState(null);
      setSecurityEvents([]);
      setError(null);
    }
  }, [user]);

  /**
   * Checks current authentication status
   */
  const checkAuth = useCallback((): boolean => {
    return !!authState?.accessToken && !!user;
  }, [authState, user]);

  /**
   * Verifies user role against required role
   */
  const hasRole = useCallback((requiredRole: UserRoleType): boolean => {
    if (!user) return false;
    return AuthService.checkAuthorization(user.role, requiredRole);
  }, [user]);

  /**
   * Continuous security monitoring process
   */
  const monitorSecurityStatus = useCallback(async (): Promise<void> => {
    if (!user || !authState) return;

    try {
      const events = await AuthService.monitorSecurityEvents(user.id);
      setSecurityEvents(events);

      // Check for critical security events
      const criticalEvents = events.filter(event => event.severity === 'critical');
      if (criticalEvents.length > 0) {
        await handleLogout();
        setError('Critical security event detected');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Security monitoring failed';
      AuthService.logAuditEvent('security_monitor_error', {
        error: errorMessage,
        userId: user.id,
        timestamp: new Date().toISOString()
      });
    }
  }, [user, authState, handleLogout]);

  /**
   * Starts security monitoring interval
   */
  const startSecurityMonitoring = useCallback(() => {
    const intervalId = setInterval(monitorSecurityStatus, SECURITY_CHECK_INTERVAL);
    return () => clearInterval(intervalId);
  }, [monitorSecurityStatus]);

  /**
   * Check security status on demand
   */
  const checkSecurityStatus = useCallback(async (): Promise<ISecurityEvent[]> => {
    await monitorSecurityStatus();
    return securityEvents;
  }, [monitorSecurityStatus, securityEvents]);

  // Initialize authentication state and security monitoring
  useEffect(() => {
    const initAuth = async () => {
      try {
        setIsLoading(true);
        const userState = await AuthService.getAuthState();
        if (userState) {
          setUser(userState);
          startSecurityMonitoring();
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Authentication initialization failed';
        setError(errorMessage);
        await handleLogout();
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, [startSecurityMonitoring, handleLogout]);

  // Context value with enhanced security features
  const contextValue: IAuthContext = {
    user,
    authState,
    isLoading,
    error,
    securityEvents,
    login: handleLogin,
    logout: handleLogout,
    checkAuth,
    hasRole,
    checkSecurityStatus
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;