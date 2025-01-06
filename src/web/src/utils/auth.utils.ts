// External imports - versions specified for security tracking
import jwtDecode from 'jwt-decode'; // v3.1.2
import { AES, mode, pad } from 'crypto-js'; // v4.1.1
import rateLimit from 'express-rate-limit'; // v6.7.0
import { TPM, SecurityLevel } from '@tpm/security'; // v2.0.0

// Internal imports
import { IUserAuth, UserRoleType } from '../interfaces/user.interface';

// Constants for authentication configuration
const TOKEN_EXPIRY_BUFFER = 900; // 15 minutes in seconds
const AUTH_STORAGE_KEY = 'tald_auth';
const ENCRYPTION_KEY = process.env.VITE_AUTH_ENCRYPTION_KEY;
const MAX_AUTH_ATTEMPTS = 5;
const RATE_LIMIT_WINDOW = 300000; // 5 minutes in milliseconds

// Initialize TPM for hardware security
const tpm = new TPM({
    securityLevel: SecurityLevel.HIGH,
    requireHardwareBackedKeys: true
});

/**
 * Parses and validates JWT token with hardware security verification
 * @param token - Raw JWT token string
 * @param hardwareToken - Hardware-specific token for TPM validation
 * @returns Parsed token data or null if invalid
 */
export const parseToken = async (
    token: string,
    hardwareToken: string
): Promise<IUserAuth | null> => {
    try {
        // Verify hardware token with TPM
        const isValidHardware = await tpm.verifyToken(hardwareToken);
        if (!isValidHardware) {
            console.error('Hardware token validation failed');
            return null;
        }

        // Decode and validate JWT
        const decoded = jwtDecode<IUserAuth>(token);
        if (!decoded?.userId || !decoded?.expiresAt) {
            console.error('Invalid token structure');
            return null;
        }

        return {
            userId: decoded.userId,
            accessToken: token,
            hardwareToken,
            expiresAt: new Date(decoded.expiresAt),
            tpmSignature: await tpm.sign(token)
        };
    } catch (error) {
        console.error('Token parsing failed:', error);
        return null;
    }
};

/**
 * Checks if a token is expired or approaching expiry
 * @param expiresAt - Token expiration timestamp
 * @returns Boolean indicating token expiration status
 */
export const isTokenExpired = (expiresAt: Date): boolean => {
    const currentTime = Math.floor(Date.now() / 1000);
    const expiryTime = Math.floor(expiresAt.getTime() / 1000);
    return currentTime >= (expiryTime - TOKEN_EXPIRY_BUFFER);
};

/**
 * Encrypts authentication token using AES-256-GCM with hardware security
 * @param token - Token string to encrypt
 * @returns Encrypted token string
 */
const encryptToken = async (token: string): Promise<string> => {
    if (!ENCRYPTION_KEY) {
        throw new Error('Encryption key not configured');
    }

    const tpmSignature = await tpm.sign(token);
    const hardwareEntropy = await tpm.getEntropy(32);
    
    return AES.encrypt(token, ENCRYPTION_KEY, {
        mode: mode.GCM,
        padding: pad.Pkcs7,
        iv: hardwareEntropy,
        additionalData: tpmSignature
    }).toString();
};

/**
 * Decrypts stored authentication token with integrity verification
 * @param encryptedToken - Encrypted token string
 * @returns Decrypted token string
 */
const decryptToken = async (encryptedToken: string): Promise<string> => {
    if (!ENCRYPTION_KEY) {
        throw new Error('Encryption key not configured');
    }

    const decrypted = AES.decrypt(encryptedToken, ENCRYPTION_KEY, {
        mode: mode.GCM,
        padding: pad.Pkcs7
    });

    const token = decrypted.toString();
    const isValid = await tpm.verify(token);
    
    if (!isValid) {
        throw new Error('Token integrity verification failed');
    }

    return token;
};

/**
 * Securely stores encrypted authentication token
 * @param authData - Authentication data to store
 */
export const storeAuthToken = async (authData: IUserAuth): Promise<void> => {
    try {
        const encryptedToken = await encryptToken(JSON.stringify(authData));
        const tpmSignature = await tpm.sign(encryptedToken);

        localStorage.setItem(AUTH_STORAGE_KEY, encryptedToken);
        localStorage.setItem(`${AUTH_STORAGE_KEY}_sig`, tpmSignature);
        
        // Set metadata for audit
        localStorage.setItem(`${AUTH_STORAGE_KEY}_meta`, JSON.stringify({
            storedAt: new Date().toISOString(),
            expiresAt: authData.expiresAt,
            hardwareVerified: true
        }));
    } catch (error) {
        console.error('Failed to store auth token:', error);
        throw error;
    }
};

/**
 * Retrieves and validates stored authentication token
 * @returns Decrypted authentication data or null if not found/invalid
 */
export const getStoredAuthToken = async (): Promise<IUserAuth | null> => {
    try {
        const encryptedToken = localStorage.getItem(AUTH_STORAGE_KEY);
        const storedSignature = localStorage.getItem(`${AUTH_STORAGE_KEY}_sig`);

        if (!encryptedToken || !storedSignature) {
            return null;
        }

        // Verify TPM signature
        const isValidSignature = await tpm.verify(encryptedToken, storedSignature);
        if (!isValidSignature) {
            console.error('Token signature verification failed');
            await clearAuthToken();
            return null;
        }

        const decryptedToken = await decryptToken(encryptedToken);
        return JSON.parse(decryptedToken);
    } catch (error) {
        console.error('Failed to retrieve auth token:', error);
        await clearAuthToken();
        return null;
    }
};

/**
 * Securely removes stored authentication token and metadata
 */
export const clearAuthToken = async (): Promise<void> => {
    try {
        // Secure wipe of sensitive data
        await tpm.secureWipe(`${AUTH_STORAGE_KEY}_sig`);
        localStorage.removeItem(AUTH_STORAGE_KEY);
        localStorage.removeItem(`${AUTH_STORAGE_KEY}_sig`);
        localStorage.removeItem(`${AUTH_STORAGE_KEY}_meta`);
    } catch (error) {
        console.error('Failed to clear auth token:', error);
        throw error;
    }
};

/**
 * Checks if current user has required role
 * @param requiredRole - Role required for access
 * @returns Boolean indicating authorization status
 */
export const hasRole = async (requiredRole: UserRoleType): Promise<boolean> => {
    try {
        const authData = await getStoredAuthToken();
        if (!authData) {
            return false;
        }

        const decoded = jwtDecode<{ role: UserRoleType }>(authData.accessToken);
        const hasRequiredRole = decoded.role === requiredRole;

        // Log access attempt for audit
        console.info('Role check:', {
            required: requiredRole,
            actual: decoded.role,
            granted: hasRequiredRole,
            timestamp: new Date().toISOString()
        });

        return hasRequiredRole;
    } catch (error) {
        console.error('Role verification failed:', error);
        return false;
    }
};

// Configure rate limiting for auth operations
export const authRateLimiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW,
    max: MAX_AUTH_ATTEMPTS,
    message: 'Too many authentication attempts, please try again later'
});