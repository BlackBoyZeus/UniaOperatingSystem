import { Request, Response, NextFunction } from 'express'; // v4.18.2
import Redis from 'ioredis'; // v5.3.2
import { TaldLogger } from '../../../utils/logger.utils';
import { RedisConfig } from '../../../config/redis.config';

// Constants for rate limiting configuration
const DEFAULT_WINDOW_MS = 60000; // 1 minute in milliseconds
const DEFAULT_MAX_REQUESTS = 100;
const RATE_LIMIT_PREFIX = 'ratelimit:';

// Endpoint-specific rate limits (requests per window)
const ENDPOINT_LIMITS = {
  'fleet/join': { windowMs: 60000, maxRequests: 10 },    // 10/min
  'fleet/sync': { windowMs: 1000, maxRequests: 20 },     // 20/sec
  'scan/upload': { windowMs: 1000, maxRequests: 30 },    // 30/sec
  'session/create': { windowMs: 60000, maxRequests: 5 }  // 5/min
} as const;

interface RateLimitOptions {
  windowMs?: number;
  maxRequests?: number;
  burstFactor?: number;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  handler?: (req: Request, res: Response) => void;
  skip?: (req: Request) => boolean;
}

interface RateLimitInfo {
  limit: number;
  current: number;
  remaining: number;
  resetTime: number;
  burstCapacity?: number;
}

/**
 * Calculates available tokens using enhanced token bucket algorithm
 */
const calculateTokens = (
  lastRefill: number,
  currentTokens: number,
  maxTokens: number,
  burstFactor: number
): number => {
  const now = Date.now();
  const timePassed = Math.max(0, now - lastRefill);
  const refillRate = maxTokens / DEFAULT_WINDOW_MS;
  const tokensToAdd = timePassed * refillRate;
  
  // Apply burst factor for temporary rate increases
  const burstCapacity = maxTokens * burstFactor;
  const calculatedTokens = Math.min(
    burstCapacity,
    currentTokens + tokensToAdd
  );

  // Apply token decay for sliding window
  const decayFactor = Math.exp(-timePassed / DEFAULT_WINDOW_MS);
  return Math.max(0, calculatedTokens * decayFactor);
};

export class RateLimiter {
  private redisClient: Redis.Cluster;
  private logger: TaldLogger;
  private options: Required<RateLimitOptions>;

  constructor(options: RateLimitOptions = {}) {
    this.options = {
      windowMs: DEFAULT_WINDOW_MS,
      maxRequests: DEFAULT_MAX_REQUESTS,
      burstFactor: 1.5,
      skipFailedRequests: false,
      keyGenerator: (req: Request) => {
        return `${req.ip}:${req.path}`;
      },
      handler: (req: Request, res: Response) => {
        res.status(429).json({
          error: 'Too Many Requests',
          retryAfter: Math.ceil(this.options.windowMs / 1000)
        });
      },
      skip: () => false,
      ...options
    };

    this.initializeRedis();
    this.initializeLogger();
  }

  private async initializeRedis(): Promise<void> {
    const redisConfig = new RedisConfig();
    this.redisClient = await redisConfig.createRedisCluster();
  }

  private initializeLogger(): void {
    this.logger = new TaldLogger({
      serviceName: 'rate-limiter',
      environment: process.env.NODE_ENV || 'development',
      enableCloudWatch: true,
      securitySettings: {
        trackAuthEvents: true,
        trackSystemIntegrity: true,
        fleetTrustThreshold: 80
      }
    });
  }

  /**
   * Express middleware for rate limiting requests
   */
  public middleware = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    if (this.options.skip(req)) {
      return next();
    }

    const key = RATE_LIMIT_PREFIX + this.options.keyGenerator(req);
    const endpoint = req.path.replace(/^\//, '');
    const limits = ENDPOINT_LIMITS[endpoint as keyof typeof ENDPOINT_LIMITS] || {
      windowMs: this.options.windowMs,
      maxRequests: this.options.maxRequests
    };

    try {
      const rateLimitInfo = await this.getRateLimit(key, limits);
      
      // Set rate limit headers
      res.setHeader('X-RateLimit-Limit', limits.maxRequests);
      res.setHeader('X-RateLimit-Remaining', Math.max(0, rateLimitInfo.remaining));
      res.setHeader('X-RateLimit-Reset', rateLimitInfo.resetTime);

      if (rateLimitInfo.remaining < 0) {
        this.logger.warn('Rate limit exceeded', {
          ip: req.ip,
          endpoint,
          current: rateLimitInfo.current,
          limit: rateLimitInfo.limit
        });

        return this.options.handler(req, res);
      }

      // Update rate limit state
      await this.redisClient.multi()
        .hincrby(key, 'tokens', -1)
        .pexpire(key, limits.windowMs)
        .exec();

      next();
    } catch (error) {
      this.logger.error('Rate limit error', error as Error, {
        ip: req.ip,
        endpoint
      });
      next(error);
    }
  };

  /**
   * Retrieves current rate limit status
   */
  private async getRateLimit(
    key: string,
    limits: { windowMs: number; maxRequests: number }
  ): Promise<RateLimitInfo> {
    const now = Date.now();
    const rateLimitData = await this.redisClient.hgetall(key);

    if (!rateLimitData.lastRefill) {
      // Initialize new rate limit entry
      await this.redisClient.hmset(key, {
        tokens: limits.maxRequests,
        lastRefill: now
      });
      await this.redisClient.pexpire(key, limits.windowMs);

      return {
        limit: limits.maxRequests,
        current: 0,
        remaining: limits.maxRequests,
        resetTime: now + limits.windowMs,
        burstCapacity: limits.maxRequests * this.options.burstFactor
      };
    }

    const tokens = calculateTokens(
      parseInt(rateLimitData.lastRefill),
      parseInt(rateLimitData.tokens),
      limits.maxRequests,
      this.options.burstFactor
    );

    return {
      limit: limits.maxRequests,
      current: limits.maxRequests - tokens,
      remaining: tokens,
      resetTime: parseInt(rateLimitData.lastRefill) + limits.windowMs,
      burstCapacity: limits.maxRequests * this.options.burstFactor
    };
  }
}

/**
 * Factory function to create rate limiter instances
 */
export const createRateLimiter = (options?: RateLimitOptions): RateLimiter => {
  return new RateLimiter(options);
};