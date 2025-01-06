import React, { useEffect, useRef, useMemo, useCallback } from 'react'; // ^18.0.0
import * as THREE from 'three'; // ^0.150.0
import { memo } from 'react'; // ^18.0.0
import { useWebGL } from '@react-three/fiber'; // ^8.0.0

import { ILidarVisualizationConfig } from '../../interfaces/lidar.interface';
import { useLidar } from '../../hooks/useLidar';
import { optimizePointCloud } from '../../utils/lidar.utils';

// Constants for WebGL and rendering optimization
const POINT_MATERIAL_SETTINGS = {
    size: 2,
    sizeAttenuation: true,
    transparent: true,
    vertexColors: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
} as const;

const CAMERA_SETTINGS = {
    fov: 75,
    near: 0.1,
    far: 5.0,
    position: [0, 0, 2],
    frustumCulled: true
} as const;

const PERFORMANCE_THRESHOLDS = {
    targetFPS: 60,
    minPointSize: 1,
    maxPointSize: 4,
    qualityLevels: ['ULTRA', 'HIGH', 'MEDIUM', 'LOW'] as const
} as const;

interface ILidarOverlayProps {
    width: number;
    height: number;
    visualConfig: ILidarVisualizationConfig;
    onPerformanceUpdate?: (metrics: { fps: number; latency: number }) => void;
}

/**
 * High-performance LiDAR visualization overlay component
 * Implements GPU-accelerated point cloud rendering with adaptive quality
 */
const LidarOverlay: React.FC<ILidarOverlayProps> = memo(({ 
    width, 
    height, 
    visualConfig,
    onPerformanceUpdate 
}) => {
    // Refs for WebGL context and rendering
    const containerRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene>(new THREE.Scene());
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const geometryRef = useRef<THREE.BufferGeometry | null>(null);
    const materialRef = useRef<THREE.PointsMaterial | null>(null);
    const pointsRef = useRef<THREE.Points | null>(null);
    const workerRef = useRef<Worker | null>(null);

    // WebGL context management
    const { gl, contextLost, contextRestored } = useWebGL();

    // LiDAR state management
    const { scanState, visualConfig: lidarConfig, handleScanUpdate, performanceMetrics } = useLidar();

    // Performance monitoring
    const fpsCounterRef = useRef<{ frames: number; lastTime: number }>({
        frames: 0,
        lastTime: performance.now()
    });

    // Initialize WebGL renderer with optimized settings
    const initializeRenderer = useCallback(() => {
        if (!containerRef.current) return;

        const renderer = new THREE.WebGLRenderer({
            antialias: false,
            powerPreference: 'high-performance',
            precision: 'highp',
            logarithmicDepthBuffer: true
        });

        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Initialize camera
        const camera = new THREE.PerspectiveCamera(
            CAMERA_SETTINGS.fov,
            width / height,
            CAMERA_SETTINGS.near,
            CAMERA_SETTINGS.far
        );
        camera.position.set(...CAMERA_SETTINGS.position);
        camera.lookAt(0, 0, 0);
        cameraRef.current = camera;

        // Initialize point cloud material
        const material = new THREE.PointsMaterial({
            ...POINT_MATERIAL_SETTINGS,
            size: visualConfig.pointSize,
            opacity: visualConfig.opacity
        });
        materialRef.current = material;

        // Initialize geometry
        const geometry = new THREE.BufferGeometry();
        geometryRef.current = geometry;

        // Create points mesh
        const points = new THREE.Points(geometry, material);
        points.frustumCulled = CAMERA_SETTINGS.frustumCulled;
        sceneRef.current.add(points);
        pointsRef.current = points;

    }, [width, height, visualConfig]);

    // Update point cloud with optimized processing
    const updatePointCloud = useCallback((points: THREE.Vector3[], quality: string) => {
        if (!geometryRef.current || !materialRef.current) return;

        const startTime = performance.now();

        // Optimize point cloud based on quality setting
        const optimizedPoints = optimizePointCloud(points, quality === 'HIGH' ? 1000000 : 500000);

        // Update geometry
        const positions = new Float32Array(optimizedPoints.length * 3);
        const colors = new Float32Array(optimizedPoints.length * 3);

        optimizedPoints.forEach((point, i) => {
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y;
            positions[i * 3 + 2] = point.z;

            // Calculate color based on depth
            const depth = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            const color = new THREE.Color();
            color.setHSL(0.6 - depth * 0.5, 1.0, 0.5);
            
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        });

        geometryRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometryRef.current.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometryRef.current.computeBoundingSphere();

        // Update performance metrics
        const processingTime = performance.now() - startTime;
        onPerformanceUpdate?.({
            fps: calculateFPS(),
            latency: processingTime
        });

    }, [onPerformanceUpdate]);

    // Handle window resize
    const handleResize = useCallback(() => {
        if (!cameraRef.current || !rendererRef.current) return;

        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(width, height);
    }, [width, height]);

    // Calculate FPS
    const calculateFPS = useCallback(() => {
        const now = performance.now();
        fpsCounterRef.current.frames++;

        if (now - fpsCounterRef.current.lastTime >= 1000) {
            const fps = fpsCounterRef.current.frames;
            fpsCounterRef.current.frames = 0;
            fpsCounterRef.current.lastTime = now;
            return fps;
        }

        return PERFORMANCE_THRESHOLDS.targetFPS;
    }, []);

    // Animation loop
    const animate = useCallback(() => {
        if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        requestAnimationFrame(animate);
        rendererRef.current.render(sceneRef.current, cameraRef.current);
        calculateFPS();
    }, [calculateFPS]);

    // Initialize component
    useEffect(() => {
        initializeRenderer();
        animate();

        // Initialize Web Worker for point cloud processing
        workerRef.current = new Worker(
            new URL('../../workers/pointcloud.worker.ts', import.meta.url)
        );

        return () => {
            if (rendererRef.current) {
                rendererRef.current.dispose();
            }
            if (geometryRef.current) {
                geometryRef.current.dispose();
            }
            if (materialRef.current) {
                materialRef.current.dispose();
            }
            if (workerRef.current) {
                workerRef.current.terminate();
            }
        };
    }, [initializeRenderer, animate]);

    // Handle scan updates
    useEffect(() => {
        if (scanState.currentScan?.points) {
            updatePointCloud(
                scanState.currentScan.points,
                performanceMetrics.quality
            );
        }
    }, [scanState.currentScan, performanceMetrics.quality, updatePointCloud]);

    // Handle resize
    useEffect(() => {
        handleResize();
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [handleResize]);

    // Handle context loss
    useEffect(() => {
        if (contextLost) {
            console.error('WebGL context lost, attempting recovery...');
            // Implement context recovery logic
        }
    }, [contextLost]);

    return (
        <div 
            ref={containerRef}
            style={{ 
                width: '100%', 
                height: '100%',
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none'
            }}
        />
    );
});

LidarOverlay.displayName = 'LidarOverlay';

export default LidarOverlay;