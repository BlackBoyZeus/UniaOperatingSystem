import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { rest } from 'msw';
import { server } from '../mocks/server';
import { useAuth } from '../../src/hooks/useAuth';
import type { IUser, UserRoleType, ISecurityEvent, IHardwareToken } from '../../src/interfaces/user.interface';

// Constants for test configuration
const MOCK_USER: IUser = {
  id: '123e4567-e89b-12d3-a456-426614174000',
  username: 'testuser',
  email: 'test@tald.unia',
  role: UserRoleType.USER,
  deviceCapabilities: {
    lidarSupported: true,
    meshNetworkSupported: true,
    vulkanVersion: '1.3',
    hardwareSecurityLevel: 'TPM2.0',
    scanningResolution: 0.01,
    maxFleetSize: 32
  },
  lastActive: new Date(),
  securityLevel: 'HIGH'
};

const MOCK_HARDWARE_TOKEN = {
  token: 'mock-hardware-token',
  tpmSignature: 'mock-tpm-signature',
  timestamp: Date.now()
};

const MOCK_SECURITY_EVENT: ISecurityEvent = {
  id: '123e4567-e89b-12d3-a456-426614174001',
  type: 'AUTH_ATTEMPT',
  severity: 'info',
  timestamp: new Date(),
  details: {
    userId: MOCK_USER.id,
    action: 'login',
    status: 'success'
  }
};

describe('useAuth Hook', () => {
  beforeEach(() => {
    // Reset MSW handlers and clear storage
    server.resetHandlers();
    localStorage.clear();
    sessionStorage.clear();

    // Reset security monitoring state
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should handle hardware token validation successfully', async () => {
    // Mock hardware token validation endpoint
    server.use(
      rest.post('/api/auth/hardware-token', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            valid: true,
            token: MOCK_HARDWARE_TOKEN
          })
        );
      })
    );

    const { result } = renderHook(() => useAuth());

    await act(async () => {
      await result.current.login(
        'testuser',
        'password123',
        MOCK_HARDWARE_TOKEN.token
      );
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(MOCK_USER);
    expect(result.current.securityLevel).toBe('HIGH');
  });

  it('should handle hardware token validation failure', async () => {
    server.use(
      rest.post('/api/auth/hardware-token', (req, res, ctx) => {
        return res(
          ctx.status(401),
          ctx.json({
            error: 'Invalid hardware token'
          })
        );
      })
    );

    const { result } = renderHook(() => useAuth());

    await expect(
      act(async () => {
        await result.current.login(
          'testuser',
          'password123',
          'invalid-token'
        );
      })
    ).rejects.toThrow('Invalid hardware token');

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('should enforce role-based access control', async () => {
    const { result } = renderHook(() => useAuth());

    // Mock successful login
    await act(async () => {
      server.use(
        rest.post('/api/auth/login', (req, res, ctx) => {
          return res(
            ctx.status(200),
            ctx.json({
              user: MOCK_USER,
              token: 'mock-jwt-token'
            })
          );
        })
      );

      await result.current.login(
        'testuser',
        'password123',
        MOCK_HARDWARE_TOKEN.token
      );
    });

    expect(result.current.hasRole(UserRoleType.USER)).toBe(true);
    expect(result.current.hasRole(UserRoleType.ADMIN)).toBe(false);
  });

  it('should handle token refresh', async () => {
    const { result } = renderHook(() => useAuth());

    // Mock initial login
    await act(async () => {
      await result.current.login(
        'testuser',
        'password123',
        MOCK_HARDWARE_TOKEN.token
      );
    });

    // Mock token refresh
    server.use(
      rest.post('/api/auth/refresh', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            token: 'new-mock-jwt-token',
            user: MOCK_USER
          })
        );
      })
    );

    // Advance timers to trigger refresh
    await act(async () => {
      jest.advanceTimersByTime(15 * 60 * 1000); // 15 minutes
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(MOCK_USER);
  });

  it('should monitor security events', async () => {
    const { result } = renderHook(() => useAuth());

    server.use(
      rest.get('/api/security/events', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            events: [MOCK_SECURITY_EVENT]
          })
        );
      })
    );

    await act(async () => {
      await result.current.monitorSecurityEvents();
    });

    expect(result.current.securityEvents).toContainEqual(MOCK_SECURITY_EVENT);
  });

  it('should handle logout and cleanup', async () => {
    const { result } = renderHook(() => useAuth());

    // Login first
    await act(async () => {
      await result.current.login(
        'testuser',
        'password123',
        MOCK_HARDWARE_TOKEN.token
      );
    });

    // Perform logout
    await act(async () => {
      await result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(result.current.securityEvents).toEqual([]);
    expect(localStorage.getItem('auth')).toBeNull();
  });

  it('should handle rate limiting', async () => {
    const { result } = renderHook(() => useAuth());

    // Attempt multiple logins rapidly
    for (let i = 0; i < 6; i++) {
      await act(async () => {
        try {
          await result.current.login(
            'testuser',
            'password123',
            MOCK_HARDWARE_TOKEN.token
          );
        } catch (error) {
          if (i >= 5) {
            expect(error).toMatchObject({
              message: expect.stringContaining('Rate limit exceeded')
            });
          }
        }
      });
    }
  });

  it('should validate TPM signatures', async () => {
    const { result } = renderHook(() => useAuth());

    server.use(
      rest.post('/api/auth/validate-tpm', (req, res, ctx) => {
        const { signature } = req.body as { signature: string };
        const isValid = signature === MOCK_HARDWARE_TOKEN.tpmSignature;

        return res(
          ctx.status(isValid ? 200 : 401),
          ctx.json({
            valid: isValid
          })
        );
      })
    );

    await act(async () => {
      const isValid = await result.current.validateHardwareToken(
        MOCK_HARDWARE_TOKEN.token,
        MOCK_HARDWARE_TOKEN.tpmSignature
      );
      expect(isValid).toBe(true);
    });
  });

  it('should maintain audit log of security events', async () => {
    const { result } = renderHook(() => useAuth());

    // Mock audit log endpoint
    server.use(
      rest.post('/api/security/audit', (req, res, ctx) => {
        return res(
          ctx.status(200),
          ctx.json({
            logged: true,
            timestamp: new Date().toISOString()
          })
        );
      })
    );

    // Perform various security-relevant actions
    await act(async () => {
      await result.current.login(
        'testuser',
        'password123',
        MOCK_HARDWARE_TOKEN.token
      );
    });

    await act(async () => {
      await result.current.logout();
    });

    // Verify audit logs
    expect(result.current.securityEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: 'AUTH_ATTEMPT'
        }),
        expect.objectContaining({
          type: 'LOGOUT'
        })
      ])
    );
  });
});