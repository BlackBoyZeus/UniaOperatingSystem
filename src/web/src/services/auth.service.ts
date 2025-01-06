import axios, { AxiosInstance, AxiosError } from 'axios'; // ^1.4.0
import jwtDecode from 'jwt-decode'; // ^3.1.2
import { createLogger, format, transports } from 'winston'; // ^3.8.0
import { TPM, TPMKey } from 'node-tpm'; // ^2.0.0

import { IUser, IUserAuth, UserRoleType } from '../interfaces/user.interface';
import { apiConfig } from '../config/api.config';

/**
 * Security event logger configuration
 */
const securityLogger = createLogger({
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.File({ filename: 'security.log' })
    ]
});

/**
 * Authentication service implementing OAuth 2.0 with hardware-backed security
 */
export class AuthService {
    private currentUser: IUserAuth | null = null;
    private readonly baseUrl: string;
    private readonly axiosInstance: AxiosInstance;
    private readonly tpm: TPM;
    private refreshTimeout?: NodeJS.Timeout;
    private rateLimitCounter: Map<string, number>;
    private readonly TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes

    constructor() {
        this.baseUrl = apiConfig.baseUrl;
        this.rateLimitCounter = new Map();
        this.tpm = new TPM();

        // Initialize axios instance with interceptors
        this.axiosInstance = axios.create(apiConfig.getRequestConfig(''));
        this.setupAxiosInterceptors();
        this.loadExistingAuth();
    }

    /**
     * Authenticates user with credentials and hardware token
     */
    public async login(
        username: string,
        password: string,
        hardwareToken: string
    ): Promise<IUserAuth> {
        try {
            // Rate limiting check
            this.checkRateLimit(username);

            // Hardware token validation
            const isValidToken = await this.validateHardwareToken(hardwareToken);
            if (!isValidToken) {
                throw new Error('Invalid hardware token');
            }

            // Authentication request
            const response = await this.axiosInstance.post<IUserAuth>(
                '/auth/login',
                {
                    username,
                    password,
                    hardwareToken,
                    tpmSignature: await this.signWithTPM(hardwareToken)
                }
            );

            // Validate and store tokens
            const authData = response.data;
            await this.validateTokens(authData);
            this.setCurrentUser(authData);

            // Setup token refresh
            this.scheduleTokenRefresh();

            // Log successful authentication
            securityLogger.info('Authentication successful', {
                userId: authData.userId,
                timestamp: new Date().toISOString()
            });

            return authData;

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
            securityLogger.error('Authentication error', {
                error: errorMessage,
                timestamp: new Date().toISOString()
            });
            throw error;
        }
    }

    /**
     * Validates hardware token using TPM
     */
    private async validateHardwareToken(hardwareToken: string): Promise<boolean> {
        try {
            // TPM validation
            const tpmKey = await this.tpm.loadKey('TALD_AUTH_KEY');
            const validation = await this.tpm.validate(hardwareToken, tpmKey);

            securityLogger.info('Hardware token validation', {
                result: validation,
                timestamp: new Date().toISOString()
            });

            return validation;
        } catch (error) {
            securityLogger.error('Hardware token validation error', {
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            });
            return false;
        }
    }

    /**
     * Refreshes access token before expiry
     */
    private async refreshToken(): Promise<IUserAuth> {
        try {
            if (!this.currentUser?.refreshToken) {
                throw new Error('No refresh token available');
            }

            const response = await this.axiosInstance.post<IUserAuth>(
                '/auth/refresh',
                {
                    refreshToken: this.currentUser.refreshToken,
                    hardwareToken: this.currentUser.hardwareToken,
                    tpmSignature: await this.signWithTPM(this.currentUser.hardwareToken)
                }
            );

            const authData = response.data;
            await this.validateTokens(authData);
            this.setCurrentUser(authData);

            securityLogger.info('Token refresh successful', {
                userId: authData.userId,
                timestamp: new Date().toISOString()
            });

            return authData;

        } catch (error) {
            securityLogger.error('Token refresh error', {
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            });
            this.handleRefreshError();
            throw error;
        }
    }

    /**
     * Signs data with TPM for hardware-backed security
     */
    private async signWithTPM(data: string): Promise<string> {
        try {
            const tpmKey = await this.tpm.loadKey('TALD_AUTH_KEY');
            return await this.tpm.sign(data, tpmKey);
        } catch (error) {
            securityLogger.error('TPM signing error', {
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            });
            throw error;
        }
    }

    /**
     * Validates JWT tokens and checks claims
     */
    private async validateTokens(authData: IUserAuth): Promise<void> {
        try {
            const accessTokenData = jwtDecode<{exp: number}>(authData.accessToken);
            const refreshTokenData = jwtDecode<{exp: number}>(authData.refreshToken);

            const now = Date.now() / 1000;
            if (accessTokenData.exp < now || refreshTokenData.exp < now) {
                throw new Error('Token expired');
            }
        } catch (error) {
            securityLogger.error('Token validation error', {
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            });
            throw error;
        }
    }

    /**
     * Sets up axios interceptors for token management
     */
    private setupAxiosInterceptors(): void {
        this.axiosInstance.interceptors.request.use(
            async (config) => {
                if (this.currentUser?.accessToken) {
                    config.headers.Authorization = `Bearer ${this.currentUser.accessToken}`;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        this.axiosInstance.interceptors.response.use(
            (response) => response,
            async (error: AxiosError) => {
                if (error.response?.status === 401) {
                    try {
                        const authData = await this.refreshToken();
                        const failedRequest = error.config;
                        if (failedRequest) {
                            failedRequest.headers.Authorization = `Bearer ${authData.accessToken}`;
                            return this.axiosInstance(failedRequest);
                        }
                    } catch (refreshError) {
                        this.handleRefreshError();
                    }
                }
                return Promise.reject(error);
            }
        );
    }

    /**
     * Implements rate limiting for authentication attempts
     */
    private checkRateLimit(identifier: string): void {
        const now = Date.now();
        const count = this.rateLimitCounter.get(identifier) || 0;
        
        if (count >= apiConfig.security.RATE_LIMIT.MAX_REQUESTS) {
            throw new Error('Rate limit exceeded');
        }

        this.rateLimitCounter.set(identifier, count + 1);
        setTimeout(() => {
            this.rateLimitCounter.delete(identifier);
        }, apiConfig.security.RATE_LIMIT.WINDOW_MS);
    }

    /**
     * Schedules token refresh before expiry
     */
    private scheduleTokenRefresh(): void {
        if (this.refreshTimeout) {
            clearTimeout(this.refreshTimeout);
        }

        if (this.currentUser?.accessToken) {
            const tokenData = jwtDecode<{exp: number}>(this.currentUser.accessToken);
            const expiresIn = (tokenData.exp * 1000) - Date.now() - this.TOKEN_REFRESH_THRESHOLD;
            
            if (expiresIn > 0) {
                this.refreshTimeout = setTimeout(() => {
                    this.refreshToken().catch(this.handleRefreshError);
                }, expiresIn);
            }
        }
    }

    /**
     * Handles token refresh errors
     */
    private handleRefreshError(): void {
        this.currentUser = null;
        if (this.refreshTimeout) {
            clearTimeout(this.refreshTimeout);
        }
        // Emit logout event
        window.dispatchEvent(new CustomEvent('auth:logout'));
    }

    /**
     * Loads existing authentication data from secure storage
     */
    private async loadExistingAuth(): Promise<void> {
        try {
            const storedAuth = localStorage.getItem('auth');
            if (storedAuth) {
                const authData: IUserAuth = JSON.parse(storedAuth);
                await this.validateTokens(authData);
                this.setCurrentUser(authData);
                this.scheduleTokenRefresh();
            }
        } catch (error) {
            securityLogger.error('Error loading stored auth', {
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString()
            });
            this.handleRefreshError();
        }
    }

    /**
     * Securely stores current user authentication data
     */
    private setCurrentUser(authData: IUserAuth): void {
        this.currentUser = authData;
        localStorage.setItem('auth', JSON.stringify(authData));
    }

    /**
     * Logs out current user and cleans up
     */
    public async logout(): Promise<void> {
        try {
            if (this.currentUser) {
                await this.axiosInstance.post('/auth/logout');
            }
        } finally {
            this.handleRefreshError();
            localStorage.removeItem('auth');
        }
    }
}

export default new AuthService();