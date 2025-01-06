// External imports with versions for security tracking
import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals'; // v29.5.0
import jwtDecode from 'jwt-decode'; // v3.1.2
import { mockTPM } from '@tpm-simulator/test'; // v1.2.0
import { SecurityAuditLogger } from '@security/audit-logger'; // v2.0.0

// Internal imports
import {
    parseToken,
    isTokenExpired,
    storeAuthToken,
    getStoredAuthToken,
    clearAuthToken,
    hasRole,
    validateHardwareToken,
    verifyTPMSignature
} from '../../src/utils/auth.utils';
import {
    IUserAuth,
    UserRoleType,
    IHardwareToken,
    ITPMSignature
} from '../../src/interfaces/user.interface';

// Mock constants
const MOCK_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
const MOCK_USER_ID = '550e8400-e29b-41d4-a716-446655440000';
const MOCK_HARDWARE_ID = 'TPM-HW-ID-001';
const MOCK_TPM_SIGNATURE = 'valid-tpm-signature-256';
const MOCK_ENCRYPTION_KEY = process.env.VITE_AUTH_ENCRYPTION_KEY;

// Mock implementations
jest.mock('@tpm/security', () => ({
    TPM: mockTPM,
    SecurityLevel: {
        HIGH: 'HIGH'
    }
}));

jest.mock('@security/audit-logger');

// Mock localStorage
const mockLocalStorage = (() => {
    let store: { [key: string]: string } = {};
    return {
        getItem: jest.fn((key: string) => store[key]),
        setItem: jest.fn((key: string, value: string) => {
            store[key] = value;
        }),
        removeItem: jest.fn((key: string) => {
            delete store[key];
        }),
        clear: jest.fn(() => {
            store = {};
        })
    };
})();

Object.defineProperty(window, 'localStorage', { value: mockLocalStorage });

describe('parseToken', () => {
    const mockAuthData: IUserAuth = {
        userId: MOCK_USER_ID,
        accessToken: MOCK_TOKEN,
        hardwareToken: MOCK_HARDWARE_ID,
        expiresAt: new Date(Date.now() + 3600000),
        tpmSignature: MOCK_TPM_SIGNATURE
    };

    beforeEach(() => {
        jest.clearAllMocks();
        mockTPM.verifyToken.mockResolvedValue(true);
        mockTPM.sign.mockResolvedValue(MOCK_TPM_SIGNATURE);
    });

    it('should successfully parse valid token with TPM signature', async () => {
        const result = await parseToken(MOCK_TOKEN, MOCK_HARDWARE_ID);
        expect(result).toEqual(mockAuthData);
        expect(mockTPM.verifyToken).toHaveBeenCalledWith(MOCK_HARDWARE_ID);
        expect(mockTPM.sign).toHaveBeenCalledWith(MOCK_TOKEN);
    });

    it('should return null for invalid hardware token', async () => {
        mockTPM.verifyToken.mockResolvedValue(false);
        const result = await parseToken(MOCK_TOKEN, 'invalid-hardware-id');
        expect(result).toBeNull();
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });

    it('should handle malformed JWT tokens', async () => {
        const result = await parseToken('invalid-token', MOCK_HARDWARE_ID);
        expect(result).toBeNull();
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });
});

describe('validateHardwareToken', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockTPM.verify.mockResolvedValue(true);
    });

    it('should validate legitimate hardware token', async () => {
        const mockHardwareToken: IHardwareToken = {
            id: MOCK_HARDWARE_ID,
            signature: MOCK_TPM_SIGNATURE,
            timestamp: Date.now()
        };

        const result = await validateHardwareToken(mockHardwareToken);
        expect(result).toBe(true);
        expect(mockTPM.verify).toHaveBeenCalled();
    });

    it('should reject expired hardware tokens', async () => {
        const mockExpiredToken: IHardwareToken = {
            id: MOCK_HARDWARE_ID,
            signature: MOCK_TPM_SIGNATURE,
            timestamp: Date.now() - 3600000 // 1 hour old
        };

        const result = await validateHardwareToken(mockExpiredToken);
        expect(result).toBe(false);
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });
});

describe('storeAuthToken', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockLocalStorage.clear();
    });

    it('should securely store token with TPM encryption', async () => {
        const mockAuthData: IUserAuth = {
            userId: MOCK_USER_ID,
            accessToken: MOCK_TOKEN,
            hardwareToken: MOCK_HARDWARE_ID,
            expiresAt: new Date(Date.now() + 3600000),
            tpmSignature: MOCK_TPM_SIGNATURE
        };

        await storeAuthToken(mockAuthData);
        
        expect(mockTPM.sign).toHaveBeenCalled();
        expect(mockLocalStorage.setItem).toHaveBeenCalledTimes(3);
        expect(SecurityAuditLogger.logAuthAttempt).toHaveBeenCalled();
    });

    it('should handle encryption failures securely', async () => {
        mockTPM.sign.mockRejectedValue(new Error('Encryption failed'));
        
        await expect(storeAuthToken({} as IUserAuth)).rejects.toThrow();
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });
});

describe('hasRole', () => {
    const mockToken = {
        role: UserRoleType.FLEET_LEADER
    };

    beforeEach(() => {
        jest.clearAllMocks();
        (jwtDecode as jest.Mock).mockReturnValue(mockToken);
    });

    it('should validate correct role authorization', async () => {
        const result = await hasRole(UserRoleType.FLEET_LEADER);
        expect(result).toBe(true);
        expect(SecurityAuditLogger.logAuthAttempt).toHaveBeenCalled();
    });

    it('should reject insufficient permissions', async () => {
        const result = await hasRole(UserRoleType.ADMIN);
        expect(result).toBe(false);
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });

    it('should handle invalid tokens securely', async () => {
        (jwtDecode as jest.Mock).mockImplementation(() => {
            throw new Error('Invalid token');
        });

        const result = await hasRole(UserRoleType.USER);
        expect(result).toBe(false);
        expect(SecurityAuditLogger.logSecurityEvent).toHaveBeenCalled();
    });
});