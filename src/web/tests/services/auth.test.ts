import { describe, test, expect, beforeAll, afterAll, beforeEach, jest } from '@jest/globals';
import { rest } from 'msw';
import { mockTPM } from '@tald/tpm-mock';

import { AuthService } from '../../src/services/auth.service';
import { IUserAuth } from '../../src/interfaces/user.interface';
import { server } from '../mocks/server';

// Mock user data for testing
const mockUser: IUserAuth = {
    userId: 'test-user-id',
    accessToken: 'mock-access-token',
    refreshToken: 'mock-refresh-token',
    hardwareToken: 'mock-hardware-token',
    tpmSignature: 'mock-signature'
};

// Mock TPM configuration
const mockTPMConfig = {
    attestationKey: 'mock-key',
    secureStorage: 'mock-storage',
    hardwareId: 'mock-hw-id'
};

describe('AuthService', () => {
    let authService: AuthService;

    beforeAll(async () => {
        // Initialize mock TPM environment
        await mockTPM.initialize(mockTPMConfig);
        server.listen();
    });

    afterAll(() => {
        server.close();
        mockTPM.cleanup();
    });

    beforeEach(() => {
        server.resetHandlers();
        localStorage.clear();
        authService = new AuthService();
        jest.clearAllMocks();
    });

    describe('Login Flow', () => {
        test('should successfully authenticate with valid credentials and hardware token', async () => {
            // Mock TPM attestation
            mockTPM.setAttestation({
                signature: 'valid-signature',
                publicKey: 'mock-public-key',
                timestamp: Date.now().toString()
            });

            // Mock successful login response
            server.use(
                rest.post('*/auth/login', (req, res, ctx) => {
                    return res(ctx.json(mockUser));
                })
            );

            const result = await authService.login(
                'testuser',
                'password123',
                'mock-hardware-token'
            );

            expect(result).toEqual(mockUser);
            expect(localStorage.getItem('auth')).toBeTruthy();
        });

        test('should fail authentication with invalid hardware token', async () => {
            mockTPM.setAttestation(null);

            await expect(
                authService.login('testuser', 'password123', 'invalid-token')
            ).rejects.toThrow('Invalid hardware token');
        });

        test('should enforce rate limiting on login attempts', async () => {
            const attempts = Array(6).fill(null).map(() => 
                authService.login('testuser', 'password123', 'mock-token')
            );

            await expect(Promise.all(attempts)).rejects.toThrow('Rate limit exceeded');
        });
    });

    describe('Token Management', () => {
        test('should automatically refresh token before expiry', async () => {
            const newToken = { ...mockUser, accessToken: 'new-access-token' };
            
            server.use(
                rest.post('*/auth/refresh', (req, res, ctx) => {
                    return res(ctx.json(newToken));
                })
            );

            // Set expired token
            const expiredToken = {
                ...mockUser,
                accessToken: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjF9.ZB6dXlw9yRyJHHHqfHKU7ZkLQ1Q'
            };
            localStorage.setItem('auth', JSON.stringify(expiredToken));

            await authService['loadExistingAuth']();
            const currentUser = await authService['getCurrentUser']();

            expect(currentUser?.accessToken).toBe(newToken.accessToken);
        });

        test('should handle token refresh failure gracefully', async () => {
            server.use(
                rest.post('*/auth/refresh', (req, res, ctx) => {
                    return res(ctx.status(401));
                })
            );

            localStorage.setItem('auth', JSON.stringify(mockUser));
            await authService['loadExistingAuth']();

            const logoutEvent = new Promise(resolve => {
                window.addEventListener('auth:logout', resolve, { once: true });
            });

            await authService['refreshToken']().catch(() => {});
            await logoutEvent;

            expect(localStorage.getItem('auth')).toBeNull();
        });
    });

    describe('Hardware Security', () => {
        test('should validate TPM attestation during login', async () => {
            const tpmSpy = jest.spyOn(mockTPM, 'validate');

            server.use(
                rest.post('*/auth/login', (req, res, ctx) => {
                    return res(ctx.json(mockUser));
                })
            );

            await authService.login('testuser', 'password123', 'mock-hardware-token');

            expect(tpmSpy).toHaveBeenCalledWith('mock-hardware-token', expect.any(Object));
        });

        test('should securely store tokens using TPM', async () => {
            const tpmStoreSpy = jest.spyOn(mockTPM, 'store');

            server.use(
                rest.post('*/auth/login', (req, res, ctx) => {
                    return res(ctx.json(mockUser));
                })
            );

            await authService.login('testuser', 'password123', 'mock-hardware-token');

            expect(tpmStoreSpy).toHaveBeenCalledWith(
                expect.stringContaining('auth'),
                expect.any(String)
            );
        });
    });

    describe('Security Monitoring', () => {
        test('should log failed authentication attempts', async () => {
            const logSpy = jest.spyOn(console, 'error');

            server.use(
                rest.post('*/auth/login', (req, res, ctx) => {
                    return res(ctx.status(401));
                })
            );

            await expect(
                authService.login('testuser', 'wrongpass', 'mock-token')
            ).rejects.toThrow();

            expect(logSpy).toHaveBeenCalledWith(
                expect.stringContaining('Authentication error')
            );
        });

        test('should track and limit failed login attempts', async () => {
            server.use(
                rest.post('*/auth/login', (req, res, ctx) => {
                    return res(ctx.status(401));
                })
            );

            const attempts = Array(5).fill(null).map(() => 
                authService.login('testuser', 'wrongpass', 'mock-token').catch(() => {})
            );

            await Promise.all(attempts);

            await expect(
                authService.login('testuser', 'password123', 'mock-token')
            ).rejects.toThrow('Rate limit exceeded');
        });

        test('should enforce token expiry checks', async () => {
            const expiredToken = {
                ...mockUser,
                accessToken: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjF9.ZB6dXlw9yRyJHHHqfHKU7ZkLQ1Q'
            };

            await expect(
                authService['validateTokens'](expiredToken)
            ).rejects.toThrow('Token expired');
        });
    });

    describe('Logout Flow', () => {
        test('should clear auth state and TPM storage on logout', async () => {
            const tpmClearSpy = jest.spyOn(mockTPM, 'clear');

            localStorage.setItem('auth', JSON.stringify(mockUser));
            await authService.logout();

            expect(localStorage.getItem('auth')).toBeNull();
            expect(tpmClearSpy).toHaveBeenCalled();
        });

        test('should handle server-side logout failures gracefully', async () => {
            server.use(
                rest.post('*/auth/logout', (req, res, ctx) => {
                    return res(ctx.status(500));
                })
            );

            localStorage.setItem('auth', JSON.stringify(mockUser));
            await authService.logout();

            expect(localStorage.getItem('auth')).toBeNull();
        });
    });
});