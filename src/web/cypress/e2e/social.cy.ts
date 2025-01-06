import { login } from '../support/commands';
import { createTestFleet } from '../support/commands';
import { regularUser } from '../fixtures/user.json';

// API routes for social features
const SOCIAL_HUB_URL = '/social';
const API_ROUTES = {
  NEARBY_PLAYERS: '/api/social/nearby',
  FRIEND_REQUESTS: '/api/social/friends/requests',
  FLEET_FORMATION: '/api/fleet/create',
  ENVIRONMENT_SHARE: '/api/environment/share',
  PERFORMANCE_METRICS: '/api/metrics/network'
} as const;

// Test timeouts and thresholds
const TEST_TIMEOUTS = {
  API_RESPONSE: 5000,
  WEBRTC_CONNECTION: 10000,
  FRIEND_REQUEST: 3000,
  FLEET_FORMATION: 15000,
  PERFORMANCE_CHECK: 2000
} as const;

const PERFORMANCE_THRESHOLDS = {
  MAX_LATENCY: 50,
  MIN_BANDWIDTH: 1000000,
  MAX_PACKET_LOSS: 0.1
} as const;

describe('Social Hub', () => {
  beforeEach(() => {
    // Configure WebRTC for testing
    cy.intercept('GET', API_ROUTES.NEARBY_PLAYERS).as('getNearbyPlayers');
    cy.intercept('POST', API_ROUTES.FLEET_FORMATION).as('createFleet');
    cy.intercept('GET', API_ROUTES.PERFORMANCE_METRICS).as('getMetrics');

    // Initialize network monitoring
    cy.window().then((win) => {
      win.performance.mark('networkStart');
    });

    // Login and visit social hub
    cy.login(regularUser.username, regularUser.auth);
    cy.visit(SOCIAL_HUB_URL);
    cy.wait('@getNearbyPlayers', { timeout: TEST_TIMEOUTS.API_RESPONSE });
  });

  it('should discover nearby players with proximity validation', () => {
    cy.get('[data-testid="nearby-players-list"]').should('be.visible');
    cy.get('[data-testid="nearby-player-card"]').should('have.length.at.least', 1);
    
    // Validate player distance display
    cy.get('[data-testid="player-distance"]').each(($distance) => {
      const distance = parseFloat($distance.text());
      expect(distance).to.be.at.most(5); // 5-meter effective range
    });
  });

  it('should manage fleet of 32 devices with performance monitoring', () => {
    // Create test fleet
    cy.get('[data-testid="create-fleet-button"]').click();
    cy.wait('@createFleet');

    // Add mock devices to fleet
    const mockDevices = Array.from({ length: 31 }, (_, i) => ({
      id: `device-${i}`,
      capabilities: regularUser.deviceCapabilities
    }));

    cy.wrap(mockDevices).each((device) => {
      cy.get('[data-testid="add-device-button"]').click();
      cy.get('[data-testid="device-id-input"]').type(device.id);
      cy.get('[data-testid="confirm-add-device"]').click();
    });

    // Validate fleet size and performance
    cy.get('[data-testid="fleet-members-list"]')
      .find('[data-testid="fleet-member"]')
      .should('have.length', 32);

    cy.wait('@getMetrics').then((interception) => {
      const metrics = interception.response.body;
      expect(metrics.latency).to.be.at.most(PERFORMANCE_THRESHOLDS.MAX_LATENCY);
      expect(metrics.bandwidth).to.be.at.least(PERFORMANCE_THRESHOLDS.MIN_BANDWIDTH);
      expect(metrics.packetLoss).to.be.at.most(PERFORMANCE_THRESHOLDS.MAX_PACKET_LOSS);
    });
  });

  it('should validate network performance for WebRTC connections', () => {
    // Create WebRTC connections
    cy.window().then((win) => {
      const rtcConfig = {
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      };
      const connections = new Array(5).fill(null).map(() => 
        new win.RTCPeerConnection(rtcConfig)
      );

      // Monitor connection states
      connections.forEach((conn, index) => {
        conn.addEventListener('connectionstatechange', () => {
          if (conn.connectionState === 'connected') {
            win.performance.mark(`connection-${index}-established`);
          }
        });
      });

      // Validate connection metrics
      cy.wait('@getMetrics', { timeout: TEST_TIMEOUTS.WEBRTC_CONNECTION })
        .then((interception) => {
          const metrics = interception.response.body;
          connections.forEach((conn) => {
            expect(conn.connectionState).to.equal('connected');
          });
          expect(metrics.p2pLatency).to.be.at.most(PERFORMANCE_THRESHOLDS.MAX_LATENCY);
        });
    });
  });

  it('should share environment data within fleet', () => {
    // Create test fleet
    cy.createTestFleet('test-fleet', 5);

    // Share environment data
    cy.fixture('mock-environment-data.json').then((envData) => {
      cy.intercept('POST', API_ROUTES.ENVIRONMENT_SHARE, envData).as('shareEnvironment');
      cy.get('[data-testid="share-environment-button"]').click();
      cy.wait('@shareEnvironment');
    });

    // Validate data synchronization
    cy.get('[data-testid="fleet-environment-status"]').should('contain', 'Synchronized');
    cy.get('[data-testid="environment-data-quality"]').should('contain', 'HIGH');
  });

  it('should handle fleet member disconnections gracefully', () => {
    cy.createTestFleet('test-fleet', 5);

    // Simulate member disconnection
    cy.window().then((win) => {
      win.dispatchEvent(new CustomEvent('fleet:memberDisconnected', {
        detail: { memberId: 'device-1' }
      }));
    });

    // Validate fleet state updates
    cy.get('[data-testid="fleet-members-list"]')
      .find('[data-testid="fleet-member"]')
      .should('have.length', 4);
    
    cy.get('[data-testid="fleet-status"]').should('contain', 'Member Disconnected');
  });

  it('should maintain fleet synchronization under network stress', () => {
    cy.createTestFleet('test-fleet', 10);

    // Simulate network conditions
    cy.window().then((win) => {
      const conditions = [
        { latency: 20, packetLoss: 0.01 },
        { latency: 40, packetLoss: 0.05 },
        { latency: 45, packetLoss: 0.08 }
      ];

      conditions.forEach((condition, index) => {
        setTimeout(() => {
          win.postMessage({
            type: 'networkCondition',
            payload: condition
          }, '*');
        }, index * 2000);
      });
    });

    // Validate fleet stability
    cy.get('[data-testid="fleet-sync-status"]', { timeout: 10000 })
      .should('contain', 'Synchronized');
    
    cy.wait('@getMetrics').then((interception) => {
      const metrics = interception.response.body;
      expect(metrics.syncSuccess).to.be.at.least(0.95);
    });
  });
});