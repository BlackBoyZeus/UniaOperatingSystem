import { defineConfig } from 'cypress';
import { beforeEach, afterEach } from './cypress/support/e2e';

// Import performance monitoring plugins
import 'cypress-performance';
import 'cypress-webrtc';

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.ts',
    specPattern: 'cypress/e2e/**/*.cy.ts',
    
    // Test execution timeouts based on technical specifications
    defaultCommandTimeout: 10000,
    requestTimeout: 15000,
    responseTimeout: 15000,
    pageLoadTimeout: 30000,
    
    // Viewport configuration for gaming display
    viewportWidth: 1920,
    viewportHeight: 1080,
    
    // Test recording and retry configuration
    video: true,
    screenshotOnFailure: true,
    retries: {
      runMode: 2,
      openMode: 0
    },

    // Environment variables for test configuration
    env: {
      // API configuration
      API_URL: 'http://localhost:8080',
      
      // WebRTC testing configuration
      WEBRTC_ENABLED: true,
      FLEET_SIZE: 32,
      NETWORK_LATENCY: 50,
      
      // LiDAR simulation configuration
      LIDAR_SIMULATION: true,
      SCAN_RATE: 30,
      POINT_CLOUD_RESOLUTION: '0.01',
      SCAN_RANGE: 5,
      
      // Performance thresholds
      FPS_TARGET: 60,
      MEMORY_LIMIT: 4096,
      GAME_TIMEOUT: 300000,
      TEST_TIMEOUT: 1200000,
      
      // Parallel test execution
      PARALLEL_INSTANCES: 4
    },

    setupNodeEvents(on, config) {
      // Register performance monitoring
      on('before:browser:launch', (browser, launchOptions) => {
        launchOptions.args.push('--js-flags=--expose-gc');
        launchOptions.args.push('--disable-gpu-vsync');
        return launchOptions;
      });

      // Configure WebRTC testing
      on('task', {
        simulateWebRTC: ({ connectionCount, latency }) => {
          return new Promise((resolve) => {
            setTimeout(() => {
              resolve({ success: true, connections: connectionCount });
            }, latency);
          });
        }
      });

      // Configure LiDAR simulation
      on('task', {
        simulateLidarScan: ({ scanRate, resolution, range }) => {
          return {
            pointCloud: new Float32Array(1000000),
            timestamp: Date.now(),
            scanRate,
            resolution,
            range
          };
        }
      });

      // Configure fleet management testing
      on('task', {
        createTestFleet: ({ size, networkLatency }) => {
          return {
            fleetId: `test-fleet-${Date.now()}`,
            members: Array(size).fill(null).map((_, i) => ({
              id: `device-${i}`,
              latency: networkLatency
            }))
          };
        }
      });

      // Configure performance monitoring
      on('task', {
        logPerformance: ({ fps, memory, latency }) => {
          return {
            timestamp: Date.now(),
            metrics: { fps, memory, latency }
          };
        }
      });

      // Configure test result reporting
      on('after:spec', (spec, results) => {
        if (results && results.video) {
          return {
            testId: `${spec.name}-${Date.now()}`,
            duration: results.duration,
            status: results.status
          };
        }
      });

      // Configure parallel test execution
      on('before:run', async () => {
        await Promise.all([
          // Initialize test environment
          config.env.API_URL && fetch(`${config.env.API_URL}/test/init`),
          // Clear test data
          config.env.API_URL && fetch(`${config.env.API_URL}/test/clear`)
        ]);
      });

      return config;
    }
  }
});