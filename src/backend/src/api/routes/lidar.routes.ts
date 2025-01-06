import { Router } from 'express'; // v4.18.2
import { container } from 'inversify'; // v6.0.1
import { rateLimit } from 'express-rate-limit'; // v6.7.0
import compression from 'compression'; // v1.7.4
import * as prometheus from 'prom-client'; // v14.2.0

import { LidarController } from '../controllers/lidar.controller';
import {
  authenticateRequest,
  authenticateDeviceRequest,
  validateFleetAccess
} from '../middlewares/auth.middleware';
import {
  validateRequest,
  validateLidarRequest
} from '../middlewares/validation.middleware';
import {
  lidarConfigSchema,
  pointCloudSchema,
  scanMetadataSchema
} from '../validators/lidar.validator';

// Constants
const LIDAR_BASE_PATH = '/api/lidar';
const SCAN_RATE_LIMIT = 30; // 30 requests per second (30Hz)
const SCAN_WINDOW_MS = 1000; // 1 second window
const SCAN_CACHE_DURATION = 300; // 5 minutes cache

// Initialize Prometheus metrics
const scanProcessingLatency = new prometheus.Histogram({
  name: 'lidar_scan_processing_latency',
  help: 'LiDAR scan processing latency in milliseconds',
  buckets: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
});

const scanQualityGauge = new prometheus.Gauge({
  name: 'lidar_scan_quality',
  help: 'LiDAR scan quality score'
});

const scanErrorCounter = new prometheus.Counter({
  name: 'lidar_scan_errors_total',
  help: 'Total number of LiDAR scan processing errors'
});

/**
 * Configures and returns Express router with enhanced LiDAR endpoints
 * including fleet awareness, monitoring, and hardware validation
 */
export function configureLidarRoutes(): Router {
  const router = Router();
  const lidarController = container.get<LidarController>(LidarController);

  // Configure rate limiting for scan processing
  const scanRateLimiter = rateLimit({
    windowMs: SCAN_WINDOW_MS,
    max: SCAN_RATE_LIMIT,
    message: 'Scan rate limit exceeded, maximum 30Hz scanning supported'
  });

  // Configure response compression
  router.use(compression({
    filter: (req, res) => {
      if (req.path.includes('/scan')) {
        return true;
      }
      return compression.filter(req, res);
    },
    level: 6
  }));

  // POST /scan - Process LiDAR scan data
  router.post('/scan',
    authenticateDeviceRequest,
    validateFleetAccess,
    scanRateLimiter,
    validateRequest(pointCloudSchema, 'body', {
      validateSecurity: true,
      validatePerformance: true,
      validateFleetContext: true
    }),
    async (req, res, next) => {
      const startTime = performance.now();
      try {
        const result = await lidarController.processScan(req, res);
        
        // Record metrics
        const processingTime = performance.now() - startTime;
        scanProcessingLatency.observe(processingTime);
        scanQualityGauge.set(result.metadata.quality);
        
        return result;
      } catch (error) {
        scanErrorCounter.inc();
        next(error);
      }
    }
  );

  // GET /scan/:scanId - Retrieve processed scan data
  router.get('/scan/:scanId',
    authenticateRequest,
    validateFleetAccess,
    async (req, res, next) => {
      try {
        const result = await lidarController.getScan(req, res);
        
        // Set caching headers
        res.set('Cache-Control', `private, max-age=${SCAN_CACHE_DURATION}`);
        res.set('ETag', `"${result.metadata.scanId}"`);
        
        return result;
      } catch (error) {
        next(error);
      }
    }
  );

  // POST /config - Update LiDAR configuration
  router.post('/config',
    authenticateDeviceRequest,
    validateRequest(lidarConfigSchema, 'body', {
      validateSecurity: true,
      validatePerformance: true
    }),
    validateLidarRequest,
    async (req, res, next) => {
      try {
        return await lidarController.updateConfig(req, res);
      } catch (error) {
        next(error);
      }
    }
  );

  // GET /health - LiDAR health check endpoint
  router.get('/health',
    authenticateDeviceRequest,
    async (req, res, next) => {
      try {
        return await lidarController.getHealth(req, res);
      } catch (error) {
        next(error);
      }
    }
  );

  // GET /metrics - LiDAR performance metrics endpoint
  router.get('/metrics',
    authenticateRequest,
    async (req, res, next) => {
      try {
        return await lidarController.getMetrics(req, res);
      } catch (error) {
        next(error);
      }
    }
  );

  return router;
}

export default configureLidarRoutes();