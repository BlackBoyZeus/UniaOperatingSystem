import { Request, Response, NextFunction } from 'express'; // v4.18.2
import createHttpError, { HttpError } from 'http-errors'; // v2.0.0
import { TaldLogger } from '../../utils/logger.utils';

// Initialize logger with security tracking and fleet awareness
const logger = new TaldLogger({
  serviceName: 'ErrorMiddleware',
  environment: process.env.NODE_ENV || 'development',
  enableCloudWatch: true,
  securitySettings: {
    trackAuthEvents: true,
    trackSystemIntegrity: true,
    fleetTrustThreshold: 80
  },
  privacySettings: {
    maskPII: true,
    sensitiveFields: ['password', 'token', 'apiKey', 'deviceId']
  },
  performanceTracking: true
});

/**
 * Enhanced custom error class with fleet and security context
 */
export class TaldError extends Error {
  public readonly statusCode: number;
  public readonly code: string;
  public readonly correlationId: string;
  public readonly metadata: Record<string, any>;
  public readonly securityContext?: Record<string, any>;
  public readonly fleetContext?: Record<string, any>;

  constructor(
    message: string,
    statusCode: number = 500,
    metadata: Record<string, any> = {},
    securityContext?: Record<string, any>,
    fleetContext?: Record<string, any>
  ) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.code = `TALD_${statusCode}`;
    this.correlationId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.metadata = metadata;
    this.securityContext = securityContext;
    this.fleetContext = fleetContext;
    Error.captureStackTrace(this, this.constructor);
  }

  /**
   * Converts error to JSON with enhanced context
   */
  toJSON(): Record<string, any> {
    return {
      error: {
        code: this.code,
        message: this.message,
        statusCode: this.statusCode,
        correlationId: this.correlationId,
        ...(this.metadata && { metadata: this.metadata }),
        ...(this.securityContext && { security: this.securityContext }),
        ...(this.fleetContext && { fleet: this.fleetContext }),
        ...(process.env.NODE_ENV === 'development' && { stack: this.stack })
      }
    };
  }
}

/**
 * Formats error response with security and fleet context
 */
const formatErrorResponse = (
  error: Error | HttpError | TaldError,
  env: string,
  fleetContext?: Record<string, any>
): Record<string, any> => {
  const isDev = env === 'development';
  const isTaldError = error instanceof TaldError;
  const isHttpError = 'statusCode' in error;

  const baseResponse = {
    error: {
      message: error.message,
      statusCode: isHttpError ? (error as HttpError).statusCode : 500,
      correlationId: isTaldError ? (error as TaldError).correlationId : `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      code: isTaldError ? (error as TaldError).code : `TALD_${isHttpError ? (error as HttpError).statusCode : 500}`,
      timestamp: new Date().toISOString()
    }
  };

  if (isDev) {
    baseResponse.error.stack = error.stack;
  }

  if (isTaldError) {
    const taldError = error as TaldError;
    if (taldError.metadata) {
      baseResponse.error.metadata = taldError.metadata;
    }
    if (taldError.securityContext) {
      baseResponse.error.security = taldError.securityContext;
    }
    if (taldError.fleetContext) {
      baseResponse.error.fleet = taldError.fleetContext;
    }
  }

  if (fleetContext) {
    baseResponse.error.fleet = {
      ...baseResponse.error.fleet,
      ...fleetContext
    };
  }

  return baseResponse;
}

/**
 * Enhanced Express error handling middleware with fleet awareness and security monitoring
 */
export const errorHandler = (
  error: Error | HttpError | TaldError,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // Determine error type and status code
  const statusCode = 'statusCode' in error ? (error as HttpError).statusCode : 500;
  const isSecurityError = statusCode >= 400 && statusCode < 404;
  const isFleetError = error.message.toLowerCase().includes('fleet');

  // Prepare fleet context if relevant
  const fleetContext = isFleetError ? {
    fleetId: req.headers['x-fleet-id'],
    fleetSize: req.headers['x-fleet-size'],
    fleetStatus: req.headers['x-fleet-status']
  } : undefined;

  // Log error with appropriate severity and context
  if (isSecurityError) {
    logger.error('Security-related error occurred', error, {
      securityContext: {
        ip: req.ip,
        path: req.path,
        method: req.method,
        headers: req.headers
      },
      fleetContext
    });
  } else if (statusCode >= 500) {
    logger.error('System error occurred', error, {
      path: req.path,
      method: req.method,
      fleetContext
    });
  } else {
    logger.warn('Client error occurred', {
      message: error.message,
      path: req.path,
      method: req.method,
      fleetContext
    });
  }

  // Format and send response
  const formattedResponse = formatErrorResponse(
    error,
    process.env.NODE_ENV || 'development',
    fleetContext
  );

  res.status(statusCode).json(formattedResponse);
};