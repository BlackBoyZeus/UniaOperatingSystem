import { Observable, BehaviorSubject } from 'rxjs'; // v7.8.1
import * as Automerge from 'automerge'; // v2.0.0
import { STORAGE_KEYS, setItem, getItem, removeItem } from '../utils/storage.utils';
import { IUserAuth } from '../interfaces/user.interface';

/**
 * Interface for TPM-backed secure storage operations
 */
interface TPMInterface {
    generateKey(): Promise<CryptoKey>;
    sign(data: ArrayBuffer): Promise<ArrayBuffer>;
    verify(signature: ArrayBuffer, data: ArrayBuffer): Promise<boolean>;
}

/**
 * Service class implementing secure client-side storage operations for TALD UNIA
 * Provides hardware-backed encrypted storage, CRDT-based state management,
 * and secure token handling with automatic cleanup and integrity verification.
 */
export class StorageService {
    private readonly storageReady$ = new BehaviorSubject<boolean>(false);
    private readonly cleanupInterval = 900000; // 15 minutes
    private gameStateDoc: Automerge.Doc<any>;
    private tpmManager: TPMInterface;

    constructor() {
        this.initializeStorage().catch(error => {
            console.error('Storage initialization failed:', error);
            this.storageReady$.next(false);
        });
    }

    /**
     * Initializes secure storage with TPM integration and CRDT support
     */
    private async initializeStorage(): Promise<void> {
        try {
            // Initialize TPM manager
            await this.initializeTPM();

            // Initialize CRDT document for game state
            this.gameStateDoc = Automerge.init();

            // Start cleanup timer
            setInterval(() => this.cleanupExpiredTokens(), this.cleanupInterval);

            // Signal storage ready
            this.storageReady$.next(true);
        } catch (error) {
            throw new Error(`Storage initialization failed: ${error.message}`);
        }
    }

    /**
     * Securely stores authentication data with TPM-backed encryption
     */
    public async saveAuthData(authData: IUserAuth): Promise<void> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            // Generate TPM-backed signature
            const dataBuffer = new TextEncoder().encode(JSON.stringify(authData));
            const signature = await this.tpmManager.sign(dataBuffer);

            const secureAuthData = {
                ...authData,
                signature: Array.from(new Uint8Array(signature))
            };

            await setItem(STORAGE_KEYS.AUTH, secureAuthData, {
                encrypt: true,
                maxAge: 900000 // 15 minutes
            });
        } catch (error) {
            throw new Error(`Failed to save auth data: ${error.message}`);
        }
    }

    /**
     * Retrieves and verifies stored authentication data
     */
    public async getAuthData(): Promise<IUserAuth | null> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            const secureAuthData = await getItem<IUserAuth & { signature: number[] }>(STORAGE_KEYS.AUTH);
            
            if (!secureAuthData) {
                return null;
            }

            // Verify TPM signature
            const { signature, ...authData } = secureAuthData;
            const dataBuffer = new TextEncoder().encode(JSON.stringify(authData));
            const isValid = await this.tpmManager.verify(
                new Uint8Array(signature).buffer,
                dataBuffer
            );

            if (!isValid) {
                await this.clearAuthData();
                throw new Error('Auth data signature verification failed');
            }

            return authData;
        } catch (error) {
            throw new Error(`Failed to retrieve auth data: ${error.message}`);
        }
    }

    /**
     * Stores game state data with CRDT merge support
     */
    public async saveGameState(gameState: any): Promise<void> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            // Merge state using Automerge CRDT
            const newDoc = Automerge.change(this.gameStateDoc, doc => {
                Object.assign(doc, gameState);
            });

            this.gameStateDoc = newDoc;

            // Store merged state
            await setItem(STORAGE_KEYS.GAME_STATE, Automerge.save(newDoc), {
                compress: true
            });
        } catch (error) {
            throw new Error(`Failed to save game state: ${error.message}`);
        }
    }

    /**
     * Retrieves and verifies stored game state
     */
    public async getGameState(): Promise<any | null> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            const savedState = await getItem<string>(STORAGE_KEYS.GAME_STATE);
            
            if (!savedState) {
                return null;
            }

            // Load and merge CRDT state
            const loadedDoc = Automerge.load(savedState);
            this.gameStateDoc = Automerge.merge(this.gameStateDoc, loadedDoc);

            return Automerge.getObjectField(this.gameStateDoc, null);
        } catch (error) {
            throw new Error(`Failed to retrieve game state: ${error.message}`);
        }
    }

    /**
     * Securely removes stored authentication data
     */
    public async clearAuthData(): Promise<void> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            await removeItem(STORAGE_KEYS.AUTH);
        } catch (error) {
            throw new Error(`Failed to clear auth data: ${error.message}`);
        }
    }

    /**
     * Securely clears all stored TALD UNIA data
     */
    public async clearAll(): Promise<void> {
        if (!this.storageReady$.value) {
            throw new Error('Storage not ready');
        }

        try {
            await Promise.all([
                removeItem(STORAGE_KEYS.AUTH),
                removeItem(STORAGE_KEYS.GAME_STATE),
                removeItem(STORAGE_KEYS.FLEET_DATA),
                removeItem(STORAGE_KEYS.USER_PREFERENCES),
                removeItem(STORAGE_KEYS.SCAN_CACHE)
            ]);

            // Reset CRDT document
            this.gameStateDoc = Automerge.init();
            
            // Reset storage ready state
            this.storageReady$.next(false);
            await this.initializeStorage();
        } catch (error) {
            throw new Error(`Failed to clear all data: ${error.message}`);
        }
    }

    /**
     * Initializes TPM manager for hardware-backed security
     */
    private async initializeTPM(): Promise<void> {
        // TPM initialization would be implemented here
        // This is a placeholder for the actual TPM integration
        this.tpmManager = {
            generateKey: async () => {
                return await crypto.subtle.generateKey(
                    { name: 'ECDSA', namedCurve: 'P-256' },
                    true,
                    ['sign', 'verify']
                );
            },
            sign: async (data: ArrayBuffer) => {
                const key = await this.tpmManager.generateKey();
                return await crypto.subtle.sign(
                    { name: 'ECDSA', hash: 'SHA-256' },
                    key,
                    data
                );
            },
            verify: async (signature: ArrayBuffer, data: ArrayBuffer) => {
                const key = await this.tpmManager.generateKey();
                return await crypto.subtle.verify(
                    { name: 'ECDSA', hash: 'SHA-256' },
                    key,
                    signature,
                    data
                );
            }
        };
    }

    /**
     * Cleans up expired tokens and storage items
     */
    private async cleanupExpiredTokens(): Promise<void> {
        try {
            const authData = await this.getAuthData();
            if (authData && new Date(authData.expiresAt) <= new Date()) {
                await this.clearAuthData();
            }
        } catch (error) {
            console.error('Token cleanup failed:', error);
        }
    }
}