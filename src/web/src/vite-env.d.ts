/// <reference types="vite/client" />

/**
 * Enhanced type definitions for TALD UNIA environment variables
 * @version 4.3.0
 */
interface ImportMetaEnv {
  /** Base API URL for TALD UNIA backend services */
  readonly VITE_API_URL: string;
  
  /** WebSocket endpoint for real-time fleet communication */
  readonly VITE_WEBSOCKET_URL: string;
  
  /** WebRTC configuration for P2P mesh networking */
  readonly VITE_WEBRTC_CONFIG: string;
  
  /** AWS region for cloud service integration */
  readonly VITE_AWS_REGION: string;
  
  /** Maximum number of devices allowed in a fleet (default: 32) */
  readonly VITE_FLEET_SIZE_LIMIT: number;
  
  /** LiDAR data update frequency in Hz (target: 30Hz) */
  readonly VITE_LIDAR_UPDATE_RATE: number;
  
  /** Maximum acceptable mesh network latency in ms (target: ≤50ms) */
  readonly VITE_MESH_LATENCY_THRESHOLD: number;
  
  /** Application mode */
  readonly MODE: 'development' | 'production';
  
  /** Development mode flag */
  readonly DEV: boolean;
  
  /** Production mode flag */
  readonly PROD: boolean;
  
  /** Server-side rendering flag */
  readonly SSR: boolean;
  
  /** Base URL for asset resolution */
  readonly BASE_URL: string;
}

/**
 * Import meta interface augmentation for Vite
 */
interface ImportMeta {
  readonly env: ImportMetaEnv;
  readonly hot: ViteHotContext;
}

/**
 * Enhanced type definitions for static image assets
 * Supports optimized loading and blur placeholders
 */
interface StaticImageData {
  /** Image source URL */
  readonly src: string;
  
  /** Image height in pixels */
  readonly height: number;
  
  /** Image width in pixels */
  readonly width: number;
  
  /** Base64 encoded blur placeholder */
  readonly blurDataURL: string;
  
  /** Image format */
  readonly format: 'png' | 'jpg' | 'webp';
}

/**
 * Type declarations for static asset imports
 */
declare module '*.svg' {
  const content: StaticImageData;
  export default content;
}

declare module '*.png' {
  const content: StaticImageData;
  export default content;
}

declare module '*.jpg' {
  const content: StaticImageData;
  export default content;
}

declare module '*.webp' {
  const content: StaticImageData;
  export default content;
}

declare module '*.glsl' {
  const content: string;
  export default content;
}

declare module '*.vert' {
  const content: string;
  export default content;
}

declare module '*.frag' {
  const content: string;
  export default content;
}