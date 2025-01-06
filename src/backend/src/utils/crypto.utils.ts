import * as crypto from 'crypto'; // ^1.0.0
import { KMS } from 'aws-sdk'; // ^2.1400.0

// Constants for cryptographic operations
const ENCRYPTION_ALGORITHM = 'aes-256-gcm';
const KEY_LENGTH = 32; // 256 bits
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;
const KEY_ROTATION_INTERVAL = 7776000000; // 90 days in milliseconds
const MAX_RETRY_ATTEMPTS = 3;

// Type definitions
interface KeyMetadata {
  createdAt: Date;
  rotationDue: Date;
  keyUsage: string;
  keyId: string;
  hardwareBacked: boolean;
}

interface EncryptionOptions {
  aad?: Buffer;
  hardwareAcceleration?: boolean;
  performanceTracking?: boolean;
}

interface EncryptionMetadata {
  timestamp: Date;
  algorithm: string;
  keyId: string;
  performanceMetrics?: {
    duration: number;
    hardwareAccelerated: boolean;
  };
}

interface DecryptionOptions {
  aad?: Buffer;
  hardwareAcceleration?: boolean;
  verifyIntegrity?: boolean;
}

interface DecryptionMetadata {
  timestamp: Date;
  algorithm: string;
  keyId: string;
  integrityVerified: boolean;
}

interface HashOptions {
  salt?: Buffer;
  iterations?: number;
  hardwareAcceleration?: boolean;
}

interface HashMetadata {
  algorithm: string;
  iterations: number;
  salt?: Buffer;
  performanceMetrics: {
    duration: number;
    hardwareAccelerated: boolean;
  };
}

interface HMACOptions {
  algorithm?: string;
  hardwareAcceleration?: boolean;
}

interface HMACMetadata {
  algorithm: string;
  keyId: string;
  timestamp: Date;
  hardwareAccelerated: boolean;
}

// Initialize AWS KMS client
const kms = new KMS({
  apiVersion: '2014-11-01',
  maxRetries: MAX_RETRY_ATTEMPTS,
  retryDelayOptions: { base: 100 }
});

/**
 * Generates a cryptographically secure key using hardware-backed random number generation
 * @param length Key length in bytes
 * @param keyUsage Intended usage of the key
 * @param retryAttempts Number of retry attempts remaining
 */
export async function generateKey(
  length: number = KEY_LENGTH,
  keyUsage: string,
  retryAttempts: number = MAX_RETRY_ATTEMPTS
): Promise<{ key: Buffer; metadata: KeyMetadata }> {
  try {
    // Request hardware-backed random bytes from AWS KMS
    const response = await kms.generateRandom({
      NumberOfBytes: length
    }).promise();

    if (!response.Plaintext) {
      throw new Error('Failed to generate random bytes from KMS');
    }

    const keyId = crypto.randomUUID();
    const metadata: KeyMetadata = {
      createdAt: new Date(),
      rotationDue: new Date(Date.now() + KEY_ROTATION_INTERVAL),
      keyUsage,
      keyId,
      hardwareBacked: true
    };

    return {
      key: Buffer.from(response.Plaintext),
      metadata
    };
  } catch (error) {
    if (retryAttempts > 0) {
      return generateKey(length, keyUsage, retryAttempts - 1);
    }
    throw new Error(`Key generation failed: ${error.message}`);
  }
}

/**
 * Encrypts data using AES-256-GCM with hardware-backed keys
 * @param data Data to encrypt
 * @param key Encryption key
 * @param options Encryption options
 */
export async function encrypt(
  data: Buffer | string,
  key: Buffer,
  options: EncryptionOptions = {}
): Promise<{ iv: Buffer; encrypted: Buffer; authTag: Buffer; metadata: EncryptionMetadata }> {
  const startTime = process.hrtime();
  
  try {
    const iv = crypto.randomBytes(IV_LENGTH);
    const cipher = crypto.createCipheriv(ENCRYPTION_ALGORITHM, key, iv, {
      authTagLength: AUTH_TAG_LENGTH
    });

    if (options.aad) {
      cipher.setAAD(options.aad);
    }

    const inputData = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const encrypted = Buffer.concat([
      cipher.update(inputData),
      cipher.final()
    ]);

    const authTag = cipher.getAuthTag();
    const [seconds, nanoseconds] = process.hrtime(startTime);

    const metadata: EncryptionMetadata = {
      timestamp: new Date(),
      algorithm: ENCRYPTION_ALGORITHM,
      keyId: crypto.createHash('sha256').update(key).digest('hex'),
      performanceMetrics: {
        duration: seconds * 1000 + nanoseconds / 1e6,
        hardwareAccelerated: options.hardwareAcceleration || false
      }
    };

    return { iv, encrypted, authTag, metadata };
  } catch (error) {
    throw new Error(`Encryption failed: ${error.message}`);
  }
}

/**
 * Decrypts AES-256-GCM encrypted data using hardware-backed keys
 * @param encrypted Encrypted data
 * @param key Decryption key
 * @param iv Initialization vector
 * @param authTag Authentication tag
 * @param options Decryption options
 */
export async function decrypt(
  encrypted: Buffer,
  key: Buffer,
  iv: Buffer,
  authTag: Buffer,
  options: DecryptionOptions = {}
): Promise<{ data: Buffer; metadata: DecryptionMetadata }> {
  try {
    const decipher = crypto.createDecipheriv(ENCRYPTION_ALGORITHM, key, iv, {
      authTagLength: AUTH_TAG_LENGTH
    });

    decipher.setAuthTag(authTag);

    if (options.aad) {
      decipher.setAAD(options.aad);
    }

    const decrypted = Buffer.concat([
      decipher.update(encrypted),
      decipher.final()
    ]);

    const metadata: DecryptionMetadata = {
      timestamp: new Date(),
      algorithm: ENCRYPTION_ALGORITHM,
      keyId: crypto.createHash('sha256').update(key).digest('hex'),
      integrityVerified: true
    };

    return { data: decrypted, metadata };
  } catch (error) {
    throw new Error(`Decryption failed: ${error.message}`);
  }
}

/**
 * Generates a cryptographic hash using SHA-512 with optional salt
 * @param data Data to hash
 * @param options Hash options
 */
export async function generateHash(
  data: string | Buffer,
  options: HashOptions = {}
): Promise<{ hash: string; metadata: HashMetadata }> {
  const startTime = process.hrtime();

  try {
    const salt = options.salt || crypto.randomBytes(16);
    const iterations = options.iterations || 100000;

    const inputData = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const hash = crypto.pbkdf2Sync(
      inputData,
      salt,
      iterations,
      64,
      'sha512'
    ).toString('hex');

    const [seconds, nanoseconds] = process.hrtime(startTime);

    const metadata: HashMetadata = {
      algorithm: 'sha512',
      iterations,
      salt,
      performanceMetrics: {
        duration: seconds * 1000 + nanoseconds / 1e6,
        hardwareAccelerated: options.hardwareAcceleration || false
      }
    };

    return { hash, metadata };
  } catch (error) {
    throw new Error(`Hash generation failed: ${error.message}`);
  }
}

/**
 * Generates an HMAC for data authentication with hardware acceleration
 * @param data Data to authenticate
 * @param key HMAC key
 * @param options HMAC options
 */
export async function generateHMAC(
  data: string | Buffer,
  key: Buffer,
  options: HMACOptions = {}
): Promise<{ hmac: string; metadata: HMACMetadata }> {
  try {
    const algorithm = options.algorithm || 'sha512';
    const hmac = crypto.createHmac(algorithm, key);
    
    const inputData = Buffer.isBuffer(data) ? data : Buffer.from(data);
    hmac.update(inputData);

    const metadata: HMACMetadata = {
      algorithm,
      keyId: crypto.createHash('sha256').update(key).digest('hex'),
      timestamp: new Date(),
      hardwareAccelerated: options.hardwareAcceleration || false
    };

    return {
      hmac: hmac.digest('hex'),
      metadata
    };
  } catch (error) {
    throw new Error(`HMAC generation failed: ${error.message}`);
  }
}