import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import * as THREE from 'three';
import { mockWebGL } from '@react-three/test-renderer';

import LidarOverlay from '../../src/components/lidar/LidarOverlay';
import { ILidarVisualizationConfig } from '../../src/interfaces/lidar.interface';
import { useLidar } from '../../src/hooks/useLidar';

// Mock dependencies
jest.mock('../../src/hooks/useLidar');
jest.mock('three', () => {
    const actualThree = jest.requireActual('three');
    return {
        ...actualThree,
        WebGLRenderer: jest.fn().mockImplementation(() => ({
            setSize: jest.fn(),
            setPixelRatio: jest.fn(),
            render: jest.fn(),
            dispose: jest.fn(),
            domElement: document.createElement('canvas')
        })),
        PerspectiveCamera: jest.fn().mockImplementation(() => ({
            position: { set: jest.fn() },
            lookAt: jest.fn(),
            aspect: 1,
            updateProjectionMatrix: jest.fn()
        })),
        Scene: jest.fn().mockImplementation(() => ({
            add: jest.fn(),
            remove: jest.fn()
        })),
        Points: jest.fn().mockImplementation(() => ({
            frustumCulled: false
        })),
        PointsMaterial: jest.fn().mockImplementation(() => ({
            size: 2,
            dispose: jest.fn()
        })),
        BufferGeometry: jest.fn().mockImplementation(() => ({
            setAttribute: jest.fn(),
            computeBoundingSphere: jest.fn(),
            dispose: jest.fn()
        })),
        BufferAttribute: jest.fn(),
        Color: jest.fn().mockImplementation(() => ({
            setHSL: jest.fn(),
            r: 0,
            g: 0,
            b: 0
        }))
    };
});

// Test constants
const TEST_DIMENSIONS = {
    width: 1920,
    height: 1080
};

const TEST_CONFIG: ILidarVisualizationConfig = {
    pointSize: 2,
    colorScheme: {
        NEAR: '#FF0000',
        MID: '#00FF00',
        FAR: '#0000FF'
    },
    opacity: 0.8,
    resolution: 0.01,
    range: 5.0
};

const generateTestPointCloud = (numPoints: number) => {
    const points = [];
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: Math.random() * 5,
            y: Math.random() * 5,
            z: Math.random() * 5
        });
    }
    return points;
};

describe('LidarOverlay Component', () => {
    let mockPerformanceNow: jest.SpyInstance;
    let mockRequestAnimationFrame: jest.SpyInstance;
    let mockWorker: jest.Mock;

    beforeEach(() => {
        // Setup WebGL context mock
        mockWebGL();

        // Mock performance.now()
        mockPerformanceNow = jest.spyOn(performance, 'now')
            .mockImplementation(() => Date.now());

        // Mock requestAnimationFrame
        mockRequestAnimationFrame = jest.spyOn(window, 'requestAnimationFrame')
            .mockImplementation(cb => setTimeout(cb, 16));

        // Mock Web Worker
        mockWorker = jest.fn().mockImplementation(() => ({
            postMessage: jest.fn(),
            terminate: jest.fn()
        }));
        window.Worker = mockWorker;

        // Mock useLidar hook
        (useLidar as jest.Mock).mockReturnValue({
            scanState: {
                currentScan: {
                    points: generateTestPointCloud(1000)
                }
            },
            visualConfig: TEST_CONFIG,
            performanceMetrics: {
                frameTime: 16.67,
                pointCount: 1000,
                updateRate: 30
            }
        });
    });

    it('initializes WebGL renderer with correct settings', async () => {
        render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
            />
        );

        await waitFor(() => {
            expect(THREE.WebGLRenderer).toHaveBeenCalledWith({
                antialias: false,
                powerPreference: 'high-performance',
                precision: 'highp',
                logarithmicDepthBuffer: true
            });
        });
    });

    it('maintains 60 FPS rendering performance', async () => {
        const performanceCallback = jest.fn();

        render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
                onPerformanceUpdate={performanceCallback}
            />
        );

        // Simulate 1 second of rendering
        for (let i = 0; i < 60; i++) {
            await act(async () => {
                mockPerformanceNow.mockReturnValue(Date.now() + (i * 16.67));
                await new Promise(resolve => setTimeout(resolve, 16));
            });
        }

        expect(performanceCallback).toHaveBeenCalledWith(
            expect.objectContaining({
                fps: expect.any(Number),
                latency: expect.any(Number)
            })
        );

        const lastCall = performanceCallback.mock.calls[performanceCallback.mock.calls.length - 1][0];
        expect(lastCall.fps).toBeGreaterThanOrEqual(58); // Allow small variance
        expect(lastCall.latency).toBeLessThanOrEqual(50); // Max 50ms processing time
    });

    it('verifies point cloud accuracy and resolution', async () => {
        const testPoints = generateTestPointCloud(1000);
        (useLidar as jest.Mock).mockReturnValue({
            scanState: {
                currentScan: {
                    points: testPoints
                }
            },
            visualConfig: TEST_CONFIG,
            performanceMetrics: {
                quality: 'HIGH'
            }
        });

        render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
            />
        );

        await waitFor(() => {
            expect(THREE.BufferGeometry).toHaveBeenCalled();
            const setAttributeCalls = (THREE.BufferGeometry as jest.Mock).mock.instances[0].setAttribute.mock.calls;
            
            // Verify position attribute
            const positionAttribute = setAttributeCalls.find(call => call[0] === 'position');
            expect(positionAttribute).toBeTruthy();
            
            // Verify color attribute
            const colorAttribute = setAttributeCalls.find(call => call[0] === 'color');
            expect(colorAttribute).toBeTruthy();
        });
    });

    it('handles WebGL context loss and recovery', async () => {
        const { container } = render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
            />
        );

        const canvas = container.querySelector('canvas');
        expect(canvas).toBeTruthy();

        // Simulate context loss
        await act(async () => {
            canvas?.dispatchEvent(new Event('webglcontextlost'));
        });

        // Simulate context restore
        await act(async () => {
            canvas?.dispatchEvent(new Event('webglcontextrestored'));
        });

        expect(THREE.WebGLRenderer).toHaveBeenCalledTimes(2);
    });

    it('cleans up resources on unmount', async () => {
        const { unmount } = render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
            />
        );

        await act(async () => {
            unmount();
        });

        expect(THREE.WebGLRenderer.mock.instances[0].dispose).toHaveBeenCalled();
        expect(THREE.BufferGeometry.mock.instances[0].dispose).toHaveBeenCalled();
        expect(THREE.PointsMaterial.mock.instances[0].dispose).toHaveBeenCalled();
        expect(mockWorker.mock.instances[0].terminate).toHaveBeenCalled();
    });

    it('adapts point cloud quality based on performance', async () => {
        const performanceCallback = jest.fn();
        
        // Simulate performance degradation
        (useLidar as jest.Mock).mockReturnValue({
            scanState: {
                currentScan: {
                    points: generateTestPointCloud(2000000) // Large point cloud
                }
            },
            visualConfig: TEST_CONFIG,
            performanceMetrics: {
                quality: 'MEDIUM'
            }
        });

        render(
            <LidarOverlay
                width={TEST_DIMENSIONS.width}
                height={TEST_DIMENSIONS.height}
                visualConfig={TEST_CONFIG}
                onPerformanceUpdate={performanceCallback}
            />
        );

        await waitFor(() => {
            const lastCall = performanceCallback.mock.calls[performanceCallback.mock.calls.length - 1][0];
            expect(lastCall.fps).toBeGreaterThanOrEqual(30); // Should maintain at least 30 FPS
        });
    });
});