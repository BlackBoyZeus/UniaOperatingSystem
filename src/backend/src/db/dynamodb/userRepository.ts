import { DynamoDB } from 'aws-sdk';
import { DocumentClient } from 'aws-sdk/clients/dynamodb';
import { injectable } from 'inversify';
import { Logger } from 'winston';
import { trace, Tracer, SpanStatusCode } from '@opentelemetry/api';
import { createHash, randomBytes, createCipheriv, createDecipheriv } from 'crypto';
import { v4 as uuidv4 } from 'uuid';

import { IUser, IUserAuth } from '../../interfaces/user.interface';

const ENCRYPTION_ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

@injectable()
export class UserRepository {
    private readonly docClient: DocumentClient;
    private readonly tableName: string;
    private readonly logger: Logger;
    private readonly tracer: Tracer;
    private readonly encryptionKey: Buffer;

    constructor(
        awsConfig: DynamoDB.ClientConfiguration,
        dbConfig: { tableName: string; encryptionKey: string },
        logger: Logger
    ) {
        // Initialize DynamoDB client with retry configuration
        this.docClient = new DocumentClient({
            ...awsConfig,
            maxRetries: 3,
            retryDelayOptions: { base: 100 },
        });

        this.tableName = dbConfig.tableName;
        this.logger = logger;
        this.tracer = trace.getTracer('user-repository');
        this.encryptionKey = Buffer.from(dbConfig.encryptionKey, 'base64');

        this.validateConfiguration();
    }

    private validateConfiguration(): void {
        if (!this.tableName || !this.encryptionKey) {
            throw new Error('Invalid repository configuration');
        }
    }

    private encryptField(data: string): { 
        encryptedData: string; 
        iv: string; 
        tag: string 
    } {
        const iv = randomBytes(IV_LENGTH);
        const cipher = createCipheriv(ENCRYPTION_ALGORITHM, this.encryptionKey, iv);
        
        let encryptedData = cipher.update(data, 'utf8', 'base64');
        encryptedData += cipher.final('base64');
        
        return {
            encryptedData,
            iv: iv.toString('base64'),
            tag: (cipher.getAuthTag()).toString('base64')
        };
    }

    private decryptField(
        encryptedData: string,
        iv: string,
        tag: string
    ): string {
        const decipher = createDecipheriv(
            ENCRYPTION_ALGORITHM,
            this.encryptionKey,
            Buffer.from(iv, 'base64')
        );
        
        decipher.setAuthTag(Buffer.from(tag, 'base64'));
        
        let decrypted = decipher.update(encryptedData, 'base64', 'utf8');
        decrypted += decipher.final('utf8');
        
        return decrypted;
    }

    public async createUser(user: IUser): Promise<IUser> {
        const span = this.tracer.startSpan('UserRepository.createUser');
        
        try {
            // Validate GDPR consent
            if (!user.gdprConsent) {
                throw new Error('GDPR consent required');
            }

            const userId = uuidv4();
            const timestamp = new Date().toISOString();

            // Encrypt sensitive fields
            const emailEncrypted = this.encryptField(user.email);
            const hardwareIdHash = createHash('sha256')
                .update(user.hardwareId)
                .digest('hex');

            const userItem = {
                id: userId,
                username: user.username,
                email: {
                    data: emailEncrypted.encryptedData,
                    iv: emailEncrypted.iv,
                    tag: emailEncrypted.tag
                },
                passwordHash: user.passwordHash,
                hardwareIdHash,
                deviceCapabilities: user.deviceCapabilities,
                gdprConsent: user.gdprConsent,
                createdAt: timestamp,
                updatedAt: timestamp,
                version: 1
            };

            await this.docClient.put({
                TableName: this.tableName,
                Item: userItem,
                ConditionExpression: 'attribute_not_exists(id)'
            }).promise();

            this.logger.info('User created successfully', {
                userId,
                username: user.username
            });

            span.setStatus({ code: SpanStatusCode.OK });
            
            // Return sanitized user data
            return {
                ...user,
                id: userId,
                createdAt: new Date(timestamp),
                updatedAt: new Date(timestamp)
            };

        } catch (error) {
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message
            });
            
            this.logger.error('Failed to create user', {
                error: error.message,
                username: user.username
            });
            
            throw error;
        } finally {
            span.end();
        }
    }

    public async getUserById(userId: string): Promise<IUser | null> {
        const span = this.tracer.startSpan('UserRepository.getUserById');
        
        try {
            const result = await this.docClient.get({
                TableName: this.tableName,
                Key: { id: userId }
            }).promise();

            if (!result.Item) {
                return null;
            }

            const user = result.Item as any;

            // Decrypt sensitive fields
            const decryptedEmail = this.decryptField(
                user.email.data,
                user.email.iv,
                user.email.tag
            );

            span.setStatus({ code: SpanStatusCode.OK });

            return {
                ...user,
                email: decryptedEmail,
                createdAt: new Date(user.createdAt),
                updatedAt: new Date(user.updatedAt)
            };

        } catch (error) {
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message
            });
            
            this.logger.error('Failed to get user', {
                error: error.message,
                userId
            });
            
            throw error;
        } finally {
            span.end();
        }
    }

    public async updateUser(userId: string, updates: Partial<IUser>): Promise<IUser> {
        const span = this.tracer.startSpan('UserRepository.updateUser');
        
        try {
            const timestamp = new Date().toISOString();
            let updateExpression = 'SET updatedAt = :timestamp';
            const expressionAttributeValues: any = {
                ':timestamp': timestamp
            };

            // Handle encrypted fields
            if (updates.email) {
                const emailEncrypted = this.encryptField(updates.email);
                updateExpression += ', email = :email';
                expressionAttributeValues[':email'] = {
                    data: emailEncrypted.encryptedData,
                    iv: emailEncrypted.iv,
                    tag: emailEncrypted.tag
                };
            }

            // Handle other fields
            Object.entries(updates).forEach(([key, value]) => {
                if (key !== 'email' && key !== 'id') {
                    updateExpression += `, ${key} = :${key}`;
                    expressionAttributeValues[`:${key}`] = value;
                }
            });

            const result = await this.docClient.update({
                TableName: this.tableName,
                Key: { id: userId },
                UpdateExpression: updateExpression,
                ExpressionAttributeValues: expressionAttributeValues,
                ReturnValues: 'ALL_NEW'
            }).promise();

            span.setStatus({ code: SpanStatusCode.OK });

            return this.getUserById(userId);

        } catch (error) {
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message
            });
            
            this.logger.error('Failed to update user', {
                error: error.message,
                userId
            });
            
            throw error;
        } finally {
            span.end();
        }
    }

    public async deleteUser(userId: string): Promise<void> {
        const span = this.tracer.startSpan('UserRepository.deleteUser');
        
        try {
            await this.docClient.delete({
                TableName: this.tableName,
                Key: { id: userId }
            }).promise();

            this.logger.info('User deleted successfully', { userId });
            span.setStatus({ code: SpanStatusCode.OK });

        } catch (error) {
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message
            });
            
            this.logger.error('Failed to delete user', {
                error: error.message,
                userId
            });
            
            throw error;
        } finally {
            span.end();
        }
    }
}