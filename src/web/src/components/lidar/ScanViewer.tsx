import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three'; // ^0.150.0
import { IPointCloud, ScanQuality } from '../../interfaces/lidar.interface';
import { useLidar } from '../../hooks/useLidar';
import { optimizePointCloud } from '../../utils/lidar.utils';
import { LIDAR_SCAN_SETTINGS, LIDAR_VISUALIZATION } from '../../constants/lidar.constants';

// Constants for visualization settings
const CAMERA_FOV = 75;
const CAMERA_NEAR = 0.1;
const CAMERA_FAR = 1000;
const POINT_SIZE = LIDAR_VISUALIZATION.POINT_SIZE;
const MAX_POINTS_PER_PARTITION = 100000;
const FLEET_SYNC_INTERVAL = 50;
const QUALITY_ADJUSTMENT_THRESHOLD = 45;

interface ScanViewerProps {
    width: number;
    height: number;
    showPerformanceMetrics?: boolean;
    enableFleetSync?: boolean;
    onQualityChange?: (quality: ScanQuality) => void;
}

interface PerformanceMetrics {
    fps: number;
    pointCount: number;
    renderTime: number;
    syncLatency: number;
}

/**
 * Enhanced ScanViewer component for real-time LiDAR visualization
 * Supports fleet synchronization and performance optimization
 */
export const ScanViewer: React.FC<ScanViewerProps> = ({
    width,
    height,
    showPerformanceMetrics = true,
    enableFleetSync = true,
    onQualityChange
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const pointCloudsRef = useRef<Map<string, THREE.Points>>(new Map());
    const animationFrameRef = useRef<number>();

    const [metrics, setMetrics] = useState<PerformanceMetrics>({
        fps: 0,
        pointCount: 0,
        renderTime: 0,
        syncLatency: 0
    });

    // Custom hook for LiDAR and fleet state management
    const { scanState, visualConfig, fleetState } = useLidar();

    // Initialize THREE.js scene with HDR support
    const initializeScene = () => {
        if (!canvasRef.current) return;

        // Create HDR-enabled renderer
        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: true,
            powerPreference: 'high-performance',
            logarithmicDepthBuffer: true
        });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        rendererRef.current = renderer;

        // Create scene with enhanced lighting
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000000, 0.001);
        sceneRef.current = scene;

        // Configure camera with dynamic FOV
        const camera = new THREE.PerspectiveCamera(
            CAMERA_FOV,
            width / height,
            CAMERA_NEAR,
            CAMERA_FAR
        );
        camera.position.set(0, 0, 5);
        cameraRef.current = camera;

        // Add ambient and directional lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(ambientLight, directionalLight);
    };

    // Update point cloud with GPU optimization
    const updatePointCloud = async (pointCloud: IPointCloud, fleetMemberId: string) => {
        if (!sceneRef.current) return;

        const startTime = performance.now();

        // Remove existing point cloud for this fleet member
        const existingPoints = pointCloudsRef.current.get(fleetMemberId);
        if (existingPoints) {
            sceneRef.current.remove(existingPoints);
            existingPoints.geometry.dispose();
        }

        // Optimize point cloud data
        const optimizedPoints = await optimizePointCloud(
            pointCloud.points,
            MAX_POINTS_PER_PARTITION
        );

        // Create geometry and material
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(optimizedPoints.length * 3);
        const colors = new Float32Array(optimizedPoints.length * 3);

        optimizedPoints.forEach((point, i) => {
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y;
            positions[i * 3 + 2] = point.z;

            // Color based on intensity and quality
            const intensity = point.intensity || 0.5;
            const qualityFactor = pointCloud.quality === ScanQuality.HIGH ? 1 : 0.7;
            colors[i * 3] = intensity * qualityFactor;
            colors[i * 3 + 1] = intensity * qualityFactor;
            colors[i * 3 + 2] = intensity * qualityFactor;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: POINT_SIZE,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: LIDAR_VISUALIZATION.OPACITY
        });

        // Create and add points to scene
        const points = new THREE.Points(geometry, material);
        sceneRef.current.add(points);
        pointCloudsRef.current.set(fleetMemberId, points);

        // Update performance metrics
        const renderTime = performance.now() - startTime;
        setMetrics(prev => ({
            ...prev,
            pointCount: optimizedPoints.length,
            renderTime
        }));

        // Notify quality changes
        if (onQualityChange && renderTime > QUALITY_ADJUSTMENT_THRESHOLD) {
            onQualityChange(ScanQuality.MEDIUM);
        }
    };

    // Animation loop with performance optimization
    const animate = () => {
        if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        const startTime = performance.now();

        // Update point cloud rotations
        pointCloudsRef.current.forEach(points => {
            points.rotation.y += 0.001;
        });

        // Render scene
        rendererRef.current.render(sceneRef.current, cameraRef.current);

        // Update FPS metrics
        const frameTime = performance.now() - startTime;
        setMetrics(prev => ({
            ...prev,
            fps: Math.round(1000 / frameTime)
        }));

        animationFrameRef.current = requestAnimationFrame(animate);
    };

    // Initialize scene
    useEffect(() => {
        initializeScene();
        animate();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            // Cleanup THREE.js resources
            pointCloudsRef.current.forEach(points => {
                points.geometry.dispose();
                (points.material as THREE.Material).dispose();
            });
            rendererRef.current?.dispose();
        };
    }, [width, height]);

    // Handle point cloud updates
    useEffect(() => {
        if (scanState.currentScan) {
            updatePointCloud(scanState.currentScan, 'local');
        }
    }, [scanState.currentScan]);

    // Handle fleet synchronization
    useEffect(() => {
        if (enableFleetSync && fleetState.isConnected) {
            const syncInterval = setInterval(() => {
                const syncStartTime = performance.now();
                // Update fleet member point clouds
                fleetState.members?.forEach(member => {
                    if (member.pointCloud) {
                        updatePointCloud(member.pointCloud, member.id);
                    }
                });
                setMetrics(prev => ({
                    ...prev,
                    syncLatency: performance.now() - syncStartTime
                }));
            }, FLEET_SYNC_INTERVAL);

            return () => clearInterval(syncInterval);
        }
    }, [enableFleetSync, fleetState.isConnected]);

    return (
        <div className="scan-viewer-container">
            <canvas ref={canvasRef} width={width} height={height} />
            {showPerformanceMetrics && (
                <div className="performance-metrics">
                    <div>FPS: {metrics.fps}</div>
                    <div>Points: {metrics.pointCount.toLocaleString()}</div>
                    <div>Render Time: {metrics.renderTime.toFixed(2)}ms</div>
                    <div>Sync Latency: {metrics.syncLatency.toFixed(2)}ms</div>
                </div>
            )}
        </div>
    );
};

export default ScanViewer;