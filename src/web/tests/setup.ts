import '@testing-library/jest-dom'; // v5.16.5
import 'whatwg-fetch'; // v3.6.2
import { beforeAll, afterAll, afterEach } from '@jest/globals'; // v29.5.0
import { WebRTCMock, RTCPeerConnectionMock } from 'webrtc-mock'; // v2.0.0
import * as Y from 'yjs'; // v13.6.0
import { server, MOCK_API_URL, MOCK_WEBSOCKET_URL } from './mocks/server';

// Global test configuration constants
export const TEST_TIMEOUT = 10000;
export const NETWORK_CONDITIONS = {
    latency: 50,
    jitter: 10,
    packetLoss: 0.1,
    bandwidth: 1000000 // 1 Mbps
};
export const MAX_FLEET_SIZE = 32;
export const CRDT_SYNC_INTERVAL = 100;

// Initialize WebRTC mock
const webrtcMock = new WebRTCMock({
    isConnected: true,
    latency: NETWORK_CONDITIONS.latency,
    jitter: NETWORK_CONDITIONS.jitter,
    packetLoss: NETWORK_CONDITIONS.packetLoss,
    bandwidth: NETWORK_CONDITIONS.bandwidth
});

// Initialize CRDT document for testing
const ydoc = new Y.Doc();
const fleetAwareness = new Y.Awareness(ydoc);

/**
 * Global test setup before all tests
 */
beforeAll(async () => {
    // Start MSW server with network simulation
    await server.listen({
        onUnhandledRequest: 'warn'
    });

    // Configure WebRTC mock environment
    global.RTCPeerConnection = RTCPeerConnectionMock;
    global.RTCSessionDescription = webrtcMock.RTCSessionDescription;
    global.RTCIceCandidate = webrtcMock.RTCIceCandidate;

    // Configure test environment variables
    process.env.VITE_API_URL = MOCK_API_URL;
    process.env.VITE_WS_URL = MOCK_WEBSOCKET_URL;
    process.env.VITE_MAX_FLEET_SIZE = MAX_FLEET_SIZE.toString();

    // Initialize CRDT test environment
    fleetAwareness.setLocalState({
        client_id: 'test-device',
        fleet_id: null,
        role: 'MEMBER',
        sync_status: 'disconnected'
    });

    // Configure fetch polyfill
    global.fetch = fetch;
});

/**
 * Global test cleanup after all tests
 */
afterAll(async () => {
    // Stop MSW server
    server.close();

    // Clean up WebRTC connections
    webrtcMock.reset();

    // Clean up CRDT documents
    ydoc.destroy();
    fleetAwareness.destroy();

    // Reset environment variables
    delete process.env.VITE_API_URL;
    delete process.env.VITE_WS_URL;
    delete process.env.VITE_MAX_FLEET_SIZE;
});

/**
 * Reset state after each test
 */
afterEach(async () => {
    // Reset MSW request handlers
    server.resetHandlers();

    // Reset network simulation conditions
    server.setNetworkProfile({
        latency: NETWORK_CONDITIONS.latency,
        jitter: NETWORK_CONDITIONS.jitter,
        packetLoss: NETWORK_CONDITIONS.packetLoss,
        bandwidth: NETWORK_CONDITIONS.bandwidth
    });

    // Reset WebRTC mock state
    webrtcMock.reset();

    // Reset CRDT state
    ydoc.transact(() => {
        for (const [key] of ydoc.share.entries()) {
            ydoc.share.delete(key);
        }
    });
    fleetAwareness.setLocalState({
        client_id: 'test-device',
        fleet_id: null,
        role: 'MEMBER',
        sync_status: 'disconnected'
    });

    // Clear any mocked timers
    jest.clearAllMocks();
    jest.clearAllTimers();

    // Clean up DOM
    document.body.innerHTML = '';
});

// Configure Jest test environment
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn()
    }))
});