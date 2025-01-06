import { AES, enc, lib } from 'crypto-js'; // v4.1.1
import { GameState } from '../types/game.types';

/**
 * Storage key constants for TALD UNIA platform
 */
export const STORAGE_KEYS = {
    AUTH: 'tald_auth',
    GAME_STATE: 'tald_game_state',
    FLEET_DATA: 'tald_fleet',
    USER_PREFERENCES: 'tald_preferences',
    SCAN_CACHE: 'tald_scan_cache',
    ENCRYPTION_METADATA: 'tald_encryption'
} as const;

/**
 * Configuration for storage operations
 */
const STORAGE_CONFIG = {
    MAX_AGE: 86400000, // 24 hours
    CLEANUP_INTERVAL: 3600000, // 1 hour
    MAX_RETRIES: 3,
    BATCH_SIZE: 100
} as const;

/**
 * Interface for storage operation options
 */
interface StorageOptions {
    encrypt?: boolean;
    maxAge?: number;
    compress?: boolean;
}

/**
 * Interface for storage metadata
 */
interface StorageMetadata {
    timestamp: number;
    encrypted: boolean;
    iv?: string;
    version: string;
    compressed?: boolean;
}

/**
 * Interface for batch operation results
 */
interface BatchResult {
    successful: number;
    failed: number;
    errors: Error[];
}

/**
 * Custom error class for storage operations
 */
export class StorageError extends Error {
    constructor(message: string, public readonly code: string) {
        super(message);
        this.name = 'StorageError';
    }
}

/**
 * Main storage management class for TALD UNIA platform
 */
export class StorageManager {
    private static instance: StorageManager;
    private encryptionKey: string;
    private cleanupInterval?: NodeJS.Timer;

    private constructor() {
        this.encryptionKey = process.env.VITE_STORAGE_ENCRYPTION_KEY || '';
        this.initializeStorage().catch(console.error);
    }

    /**
     * Get singleton instance of StorageManager
     */
    public static getInstance(): StorageManager {
        if (!StorageManager.instance) {
            StorageManager.instance = new StorageManager();
        }
        return StorageManager.instance;
    }

    /**
     * Initialize storage system with cleanup and monitoring
     */
    private async initializeStorage(): Promise<void> {
        if (!this.isStorageAvailable()) {
            throw new StorageError('Local storage is not available', 'STORAGE_UNAVAILABLE');
        }

        // Setup storage event listeners for cross-tab synchronization
        window.addEventListener('storage', this.handleStorageEvent.bind(this));

        // Initialize cleanup interval
        this.cleanupInterval = setInterval(
            () => this.cleanupExpiredItems(),
            STORAGE_CONFIG.CLEANUP_INTERVAL
        );

        // Monitor storage quota
        await this.monitorStorageQuota();
    }

    /**
     * Store an item with encryption and metadata
     */
    public async setItem<T>(key: string, value: T, options: StorageOptions = {}): Promise<void> {
        try {
            const metadata: StorageMetadata = {
                timestamp: Date.now(),
                encrypted: !!options.encrypt,
                version: '1.0',
                compressed: !!options.compress
            };

            let processedValue: string;
            if (options.encrypt) {
                const iv = lib.WordArray.random(16);
                metadata.iv = iv.toString();
                processedValue = this.encrypt(JSON.stringify(value), iv);
            } else {
                processedValue = JSON.stringify(value);
            }

            const storageValue = JSON.stringify({
                data: processedValue,
                metadata
            });

            await this.retryOperation(() => {
                localStorage.setItem(key, storageValue);
            });

            // Emit storage event for cross-tab sync
            window.dispatchEvent(new StorageEvent('storage', {
                key,
                newValue: storageValue
            }));
        } catch (error) {
            throw new StorageError(
                `Failed to store item: ${error.message}`,
                'STORAGE_WRITE_ERROR'
            );
        }
    }

    /**
     * Retrieve and decrypt an item from storage
     */
    public async getItem<T>(key: string): Promise<T | null> {
        try {
            const rawValue = localStorage.getItem(key);
            if (!rawValue) return null;

            const { data, metadata } = JSON.parse(rawValue);
            
            // Check expiration
            if (this.isExpired(metadata.timestamp)) {
                await this.removeItem(key);
                return null;
            }

            let processedData: T;
            if (metadata.encrypted && metadata.iv) {
                const decrypted = this.decrypt(data, metadata.iv);
                processedData = JSON.parse(decrypted);
            } else {
                processedData = JSON.parse(data);
            }

            return processedData;
        } catch (error) {
            throw new StorageError(
                `Failed to retrieve item: ${error.message}`,
                'STORAGE_READ_ERROR'
            );
        }
    }

    /**
     * Perform batch operations on storage items
     */
    public async batchOperation(operations: Array<{
        type: 'set' | 'get' | 'remove';
        key: string;
        value?: any;
        options?: StorageOptions;
    }>): Promise<BatchResult> {
        const result: BatchResult = {
            successful: 0,
            failed: 0,
            errors: []
        };

        for (let i = 0; i < operations.length; i += STORAGE_CONFIG.BATCH_SIZE) {
            const batch = operations.slice(i, i + STORAGE_CONFIG.BATCH_SIZE);
            await Promise.all(batch.map(async operation => {
                try {
                    switch (operation.type) {
                        case 'set':
                            await this.setItem(operation.key, operation.value, operation.options);
                            break;
                        case 'get':
                            await this.getItem(operation.key);
                            break;
                        case 'remove':
                            await this.removeItem(operation.key);
                            break;
                    }
                    result.successful++;
                } catch (error) {
                    result.failed++;
                    result.errors.push(error);
                }
            }));
        }

        return result;
    }

    /**
     * Remove an item from storage
     */
    private async removeItem(key: string): Promise<void> {
        await this.retryOperation(() => {
            localStorage.removeItem(key);
        });
    }

    /**
     * Encrypt data using AES-256-GCM
     */
    private encrypt(data: string, iv: lib.WordArray): string {
        return AES.encrypt(data, this.encryptionKey, {
            iv,
            mode: lib.mode.GCM,
            padding: lib.pad.Pkcs7
        }).toString();
    }

    /**
     * Decrypt data using AES-256-GCM
     */
    private decrypt(encryptedData: string, iv: string): string {
        const decrypted = AES.decrypt(encryptedData, this.encryptionKey, {
            iv: enc.Hex.parse(iv),
            mode: lib.mode.GCM,
            padding: lib.pad.Pkcs7
        });
        return decrypted.toString(enc.Utf8);
    }

    /**
     * Check if storage is available
     */
    private isStorageAvailable(): boolean {
        try {
            const test = '__storage_test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Handle storage events for cross-tab synchronization
     */
    private handleStorageEvent(event: StorageEvent): void {
        if (!event.key || !event.newValue) return;
        // Implement specific sync logic if needed
    }

    /**
     * Monitor storage quota usage
     */
    private async monitorStorageQuota(): Promise<void> {
        if (navigator.storage && navigator.storage.estimate) {
            const { usage, quota } = await navigator.storage.estimate();
            if (usage && quota && (usage / quota > 0.9)) {
                console.warn('Storage quota usage above 90%');
            }
        }
    }

    /**
     * Clean up expired items
     */
    private async cleanupExpiredItems(): Promise<void> {
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key) {
                const rawValue = localStorage.getItem(key);
                if (rawValue) {
                    try {
                        const { metadata } = JSON.parse(rawValue);
                        if (this.isExpired(metadata.timestamp)) {
                            await this.removeItem(key);
                        }
                    } catch {
                        // Skip invalid items
                    }
                }
            }
        }
    }

    /**
     * Check if data is expired
     */
    private isExpired(timestamp: number): boolean {
        return Date.now() - timestamp > STORAGE_CONFIG.MAX_AGE;
    }

    /**
     * Retry operation with exponential backoff
     */
    private async retryOperation(operation: () => void): Promise<void> {
        let retries = 0;
        while (retries < STORAGE_CONFIG.MAX_RETRIES) {
            try {
                operation();
                return;
            } catch (error) {
                retries++;
                if (retries === STORAGE_CONFIG.MAX_RETRIES) {
                    throw error;
                }
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, retries) * 100));
            }
        }
    }
}