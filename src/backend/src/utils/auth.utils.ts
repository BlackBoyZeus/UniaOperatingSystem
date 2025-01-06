/**
 * @fileoverview Core authentication utilities with hardware-backed security
 * @version 1.0.0
 * @license MIT
 */

import * as jwt from 'jsonwebtoken'; // ^9.0.0
import { KMS } from 'aws-sdk'; // ^2.1400.0
import * as TPM2 from 'node-tpm2'; // ^2.0.0
import { IUser } from '../interfaces/user.interface';
import { generateKey } from './crypto.utils';

// Environment variables and constants
const JWT_SECRET = process.env.JWT_SECRET;
const TOKEN_EXPIRY = 900; // 15 minutes
const REFRESH_TOKEN_EXPIRY = 604800; // 7 days
const HARDWARE_TOKEN_EXPIRY = 7200; // 2 hours
const KMS_KEY_ID = process.env.KMS_KEY_ID;
const TPM_DEVICE_PATH = process.env.TPM_DEVICE_PATH;

// Initialize AWS KMS client
const kms = new KMS({
  apiVersion: '2014-11-01',
  maxRetries: 3
});

// Initialize TPM2 client
const tpm = new TPM2.TPM2({
  devicePath: TPM_DEVICE_PATH
});

/**
 * Generates a JWT token with hardware-backed security and KMS integration
 * @param payload Token payload data
 * @param options JWT signing options
 * @returns Promise<string> Generated JWT token with hardware attestation
 */
export async function generateToken(
  payload: Record<string, any>,
  options: jwt.SignOptions
): Promise<string> {
  try {
    // Generate hardware-backed key using AWS KMS
    const { key, metadata } = await generateKey(32, 'token-signing');

    // Verify TPM attestation status
    const tpmAttestation = await tpm.getAttestationKey();
    
    // Add standard claims
    const tokenPayload = {
      ...payload,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + (options.expiresIn as number || TOKEN_EXPIRY),
      iss: 'tald-unia-platform',
      // Add hardware attestation claims
      hwt: {
        kid: metadata.keyId,
        tpm: tpmAttestation.publicKey,
        sig: await tpm.sign(metadata.keyId)
      }
    };

    // Sign token with KMS-generated key
    const signParams = {
      KeyId: KMS_KEY_ID,
      Message: Buffer.from(JSON.stringify(tokenPayload)),
      SigningAlgorithm: 'RSASSA_PKCS1_V1_5_SHA_256'
    };
    
    const signature = await kms.sign(signParams).promise();
    
    return jwt.sign(tokenPayload, key, {
      ...options,
      algorithm: 'RS256',
      jwtid: metadata.keyId
    });
  } catch (error) {
    throw new Error(`Token generation failed: ${error.message}`);
  }
}

/**
 * Verifies JWT token authenticity, expiration, and hardware attestation
 * @param token JWT token to verify
 * @returns Promise<Record<string, any>> Decoded and verified token payload
 */
export async function verifyToken(token: string): Promise<Record<string, any>> {
  try {
    // Decode token without verification first to get key ID
    const decoded = jwt.decode(token, { complete: true });
    if (!decoded) throw new Error('Invalid token format');

    // Verify token signature using KMS
    const verifyParams = {
      KeyId: KMS_KEY_ID,
      Message: Buffer.from(JSON.stringify(decoded.payload)),
      Signature: Buffer.from(decoded.signature, 'base64'),
      SigningAlgorithm: 'RSASSA_PKCS1_V1_5_SHA_256'
    };

    await kms.verify(verifyParams).promise();

    // Verify hardware attestation claims
    const hwt = decoded.payload.hwt;
    if (!hwt) throw new Error('Missing hardware attestation');

    const tpmVerification = await tpm.verify(hwt.kid, hwt.sig);
    if (!tpmVerification) throw new Error('Invalid hardware attestation');

    // Verify token expiration
    if (decoded.payload.exp < Math.floor(Date.now() / 1000)) {
      throw new Error('Token expired');
    }

    return decoded.payload;
  } catch (error) {
    throw new Error(`Token verification failed: ${error.message}`);
  }
}

/**
 * Generates a hardware-specific token with TPM 2.0 attestation
 * @param user User object containing hardware ID
 * @returns Promise<string> Hardware-specific authentication token
 */
export async function generateHardwareToken(user: IUser): Promise<string> {
  try {
    // Verify TPM presence and status
    const tpmStatus = await tpm.getStatus();
    if (!tpmStatus.enabled) throw new Error('TPM not enabled');

    // Generate hardware-specific key with TPM
    const tpmKey = await tpm.createKey({
      parent: 'owner',
      type: 'rsa',
      attributes: ['sign', 'decrypt']
    });

    // Create token with device and TPM claims
    const payload = {
      sub: user.id,
      hwid: user.hardwareId,
      tpm: {
        key: tpmKey.public,
        pcr: await tpm.getPCRValues([0, 1, 2, 3]),
        aik: await tpm.getAttestationKey()
      },
      exp: Math.floor(Date.now() / 1000) + HARDWARE_TOKEN_EXPIRY
    };

    // Sign token with TPM-backed key
    const signature = await tpm.sign(JSON.stringify(payload), tpmKey.handle);

    return jwt.sign(
      { ...payload, sig: signature },
      JWT_SECRET,
      { algorithm: 'RS256', expiresIn: HARDWARE_TOKEN_EXPIRY }
    );
  } catch (error) {
    throw new Error(`Hardware token generation failed: ${error.message}`);
  }
}

/**
 * Verifies hardware-specific authentication token with TPM attestation
 * @param token Hardware authentication token
 * @param hardwareId Expected hardware ID
 * @returns Promise<boolean> True if token is valid with verified hardware attestation
 */
export async function verifyHardwareToken(
  token: string,
  hardwareId: string
): Promise<boolean> {
  try {
    // Decode hardware token
    const decoded = jwt.verify(token, JWT_SECRET) as Record<string, any>;

    // Verify hardware ID match
    if (decoded.hwid !== hardwareId) {
      throw new Error('Hardware ID mismatch');
    }

    // Verify TPM signature
    const validSignature = await tpm.verify(
      decoded.tpm.key,
      decoded.sig,
      JSON.stringify({
        ...decoded,
        sig: undefined
      })
    );

    if (!validSignature) {
      throw new Error('Invalid TPM signature');
    }

    // Verify TPM attestation claims
    const currentPCR = await tpm.getPCRValues([0, 1, 2, 3]);
    const pcrMatch = decoded.tpm.pcr.every(
      (value: string, index: number) => value === currentPCR[index]
    );

    if (!pcrMatch) {
      throw new Error('PCR values mismatch');
    }

    return true;
  } catch (error) {
    throw new Error(`Hardware token verification failed: ${error.message}`);
  }
}