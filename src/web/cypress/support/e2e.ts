// External imports with versions for security tracking
import 'cypress'; // ^12.0.0
import '@testing-library/webrtc-mock'; // ^2.0.0
import '@cypress/code-coverage'; // ^3.10.0

// Internal imports
import './commands';

// Global constants from technical specifications
const TEST_TIMEOUT = 60000;
const VIEWPORT_WIDTH = 1920;
const VIEWPORT_HEIGHT = 1080;
const MAX_FLEET_SIZE = 32;
const LIDAR_SCAN_RATE = 30;
const TARGET_FPS = 60;
const MAX_LATENCY = 50;
const MEMORY_THRESHOLD = 4096; // 4GB in MB

/**
 * Configure global test environment
 */
Cypress.config({
  defaultCommandTimeout: 10000,
  requestTimeout: 15000,
  responseTimeout: 15000,
  pageLoadTimeout: 30000,
  viewportWidth: VIEWPORT_WIDTH,
  viewportHeight: VIEWPORT_HEIGHT,
  retries: {
    runMode: 2,
    openMode: 0
  },
  video: true,
  screenshotOnFailure: true
});

/**
 * Global test setup that runs before each test
 */
beforeEach(() => {
  // Clear test state
  cy.clearLocalStorage();
  cy.clearCookies();

  // Reset viewport
  cy.viewport(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);

  // Initialize WebRTC mock with fleet support
  cy.window().then((win) => {
    win.WebRTC = {
      maxConnections: MAX_FLEET_SIZE,
      simulateLatency: true,
      targetLatency: MAX_LATENCY,
      connectionTimeout: TEST_TIMEOUT,
      peerConfig: {
        iceServers: [{
          urls: ['stun:stun1.l.google.com:19302']
        }]
      }
    };
  });

  // Configure LiDAR simulation
  cy.window().then((win) => {
    win.LiDAR = {
      scanRate: LIDAR_SCAN_RATE,
      resolution: 0.01,
      range: 5.0,
      simulatePointCloud: true,
      maxPoints: 1000000
    };
  });

  // Initialize performance monitoring
  cy.window().then((win) => {
    win.performance.mark('testStart');
    win.performanceMetrics = {
      fps: [],
      memory: [],
      latency: [],
      startTime: Date.now()
    };
  });

  // Set up network interception
  cy.intercept('/api/**', (req) => {
    req.on('response', (res) => {
      res.headers['x-response-time'] = '0';
    });
  });

  // Configure OAuth mock responses
  cy.intercept('POST', '**/oauth/token', {
    statusCode: 200,
    body: {
      access_token: 'test-token',
      refresh_token: 'test-refresh-token',
      expires_in: 3600
    }
  });

  // Initialize fleet state management
  cy.window().then((win) => {
    win.fleetState = {
      members: [],
      maxSize: MAX_FLEET_SIZE,
      connections: new Map(),
      latencies: new Map()
    };
  });
});

/**
 * Global test cleanup that runs after each test
 */
afterEach(() => {
  // Clean up WebRTC connections
  cy.window().then((win) => {
    win.fleetState?.connections.forEach((conn) => conn.close());
    win.fleetState?.connections.clear();
  });

  // Clear LiDAR simulation data
  cy.window().then((win) => {
    win.LiDAR = null;
  });

  // Generate performance report
  cy.window().then((win) => {
    win.performance.mark('testEnd');
    win.performance.measure('testDuration', 'testStart', 'testEnd');

    const metrics = win.performanceMetrics;
    const avgFps = metrics.fps.reduce((a, b) => a + b, 0) / metrics.fps.length;
    const avgMemory = metrics.memory.reduce((a, b) => a + b, 0) / metrics.memory.length;
    const avgLatency = metrics.latency.reduce((a, b) => a + b, 0) / metrics.latency.length;

    cy.log('Performance Metrics', {
      averageFPS: avgFps,
      targetFPS: TARGET_FPS,
      averageMemory: `${avgMemory}MB`,
      memoryThreshold: `${MEMORY_THRESHOLD}MB`,
      averageLatency: `${avgLatency}ms`,
      latencyThreshold: `${MAX_LATENCY}ms`,
      testDuration: Date.now() - metrics.startTime
    });

    // Assert performance requirements
    expect(avgFps).to.be.gte(TARGET_FPS * 0.9); // Allow 10% deviation
    expect(avgMemory).to.be.lte(MEMORY_THRESHOLD);
    expect(avgLatency).to.be.lte(MAX_LATENCY);
  });

  // Clean up fleet state
  cy.window().then((win) => {
    win.fleetState = null;
  });

  // Reset test environment
  cy.window().then((win) => {
    win.WebRTC = null;
    win.performanceMetrics = null;
  });

  // Clear intercepted requests
  cy.intercept('**', null);
});

// Prevent TypeScript errors on custom commands
declare global {
  namespace Cypress {
    interface Chainable {
      login(username: string, password: string, hardwareToken: string): Chainable<void>;
      createTestFleet(fleetId: string, size: number): Chainable<void>;
      joinTestFleet(fleetId: string): Chainable<void>;
      startTestGame(gameId: string): Chainable<void>;
      simulateLidarData(scanData: object): Chainable<void>;
      monitorPerformance(thresholds: object): Chainable<void>;
    }
  }
}