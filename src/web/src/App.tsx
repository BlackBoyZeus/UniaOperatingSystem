import React, { useEffect, useMemo } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import DashboardLayout from './layouts/DashboardLayout';

// Gaming-optimized query client configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5 seconds
      cacheTime: 300000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always',
      suspense: true,
    }
  }
});

// Security configuration for route protection
const securityConfig = {
  tokenValidationInterval: 5000, // 5 seconds
  auditLogLevel: 'detailed',
  hardwareCheckInterval: 10000, // 10 seconds
};

/**
 * Enhanced root application component with hardware security,
 * HDR support, and gaming optimizations
 */
const App: React.FC = React.memo(() => {
  // Monitor hardware security status
  useEffect(() => {
    const securityInterval = setInterval(() => {
      AuthProvider.monitorSecurityEvents().catch(console.error);
    }, securityConfig.hardwareCheckInterval);

    return () => clearInterval(securityInterval);
  }, []);

  // Memoized route configuration with security levels
  const protectedRoutes = useMemo(() => [
    {
      path: '/',
      element: (
        <DashboardLayout>
          <React.Suspense fallback={<div>Loading...</div>}>
            {React.lazy(() => import('./pages/Dashboard'))}
          </React.Suspense>
        </DashboardLayout>
      ),
      security: {
        requiresHardwareToken: true,
        minimumRole: 'user',
        auditLevel: 'high'
      }
    },
    {
      path: '/fleet',
      element: (
        <DashboardLayout>
          <React.Suspense fallback={<div>Loading...</div>}>
            {React.lazy(() => import('./pages/Fleet'))}
          </React.Suspense>
        </DashboardLayout>
      ),
      security: {
        requiresHardwareToken: true,
        minimumRole: 'fleetLeader',
        auditLevel: 'high'
      }
    },
    {
      path: '/game',
      element: (
        <DashboardLayout>
          <React.Suspense fallback={<div>Loading...</div>}>
            {React.lazy(() => import('./pages/Game'))}
          </React.Suspense>
        </DashboardLayout>
      ),
      security: {
        requiresHardwareToken: true,
        minimumRole: 'user',
        auditLevel: 'critical'
      }
    }
  ], []);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <BrowserRouter>
            <Routes>
              {protectedRoutes.map(({ path, element, security }) => (
                <Route
                  key={path}
                  path={path}
                  element={
                    <RequireAuth
                      minimumRole={security.minimumRole}
                      requiresHardwareToken={security.requiresHardwareToken}
                      auditLevel={security.auditLevel}
                    >
                      {element}
                    </RequireAuth>
                  }
                />
              ))}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
});

/**
 * Enhanced authentication wrapper with hardware security validation
 */
interface RequireAuthProps {
  children: React.ReactNode;
  minimumRole: string;
  requiresHardwareToken: boolean;
  auditLevel: string;
}

const RequireAuth: React.FC<RequireAuthProps> = ({
  children,
  minimumRole,
  requiresHardwareToken,
  auditLevel
}) => {
  const { user, hasRole, validateHardwareToken } = AuthProvider.useAuthContext();

  useEffect(() => {
    if (requiresHardwareToken) {
      validateHardwareToken().catch(console.error);
    }
  }, [requiresHardwareToken, validateHardwareToken]);

  if (!user || (minimumRole && !hasRole(minimumRole))) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

App.displayName = 'App';

export default App;