import React from 'react';
import ReactDOM from 'react-dom/client';
import { ErrorBoundary } from 'react-error-boundary';
import { onCLS, onFID, onLCP, onFCP, onTTFB } from 'web-vitals';

import App from './App';
import './styles/global.css';

/**
 * Initializes HDR display capabilities and color space configuration
 * @returns Promise<boolean> indicating whether HDR is supported and enabled
 */
async function initializeHDR(): Promise<boolean> {
  try {
    // Check for HDR support
    const hdrSupported = window.matchMedia('(dynamic-range: high)').matches;
    
    // Check for P3 color space support
    const p3Supported = CSS.supports('color', 'color(display-p3 0 0 0)');
    
    if (hdrSupported) {
      document.documentElement.style.setProperty('--color-gamut', 'p3');
      document.documentElement.style.setProperty('--dynamic-range', 'high');
    }

    // Configure color space
    if (p3Supported) {
      document.documentElement.style.setProperty('--color-space', 'display-p3');
    }

    return hdrSupported && p3Supported;
  } catch (error) {
    console.error('HDR initialization failed:', error);
    return false;
  }
}

/**
 * Initializes performance monitoring and optimization features
 */
function setupPerformanceMonitoring(): void {
  // Core Web Vitals monitoring
  onCLS(metric => console.log('CLS:', metric.value));
  onFID(metric => console.log('FID:', metric.value));
  onLCP(metric => console.log('LCP:', metric.value));
  onFCP(metric => console.log('FCP:', metric.value));
  onTTFB(metric => console.log('TTFB:', metric.value));

  // Frame timing monitoring
  let lastFrameTime = performance.now();
  const frameCallback = () => {
    const currentTime = performance.now();
    const frameDuration = currentTime - lastFrameTime;
    
    if (frameDuration > 16.67) { // Below 60fps
      document.documentElement.style.setProperty('--animation-duration', '500ms');
    } else {
      document.documentElement.style.setProperty('--animation-duration', '300ms');
    }
    
    lastFrameTime = currentTime;
    requestAnimationFrame(frameCallback);
  };
  requestAnimationFrame(frameCallback);

  // Power usage monitoring
  if ('getBattery' in navigator) {
    (navigator as any).getBattery().then((battery: any) => {
      const updatePowerMode = () => {
        const powerSaveMode = battery.charging ? 'HIGH_PERFORMANCE' : 'POWER_SAVER';
        document.documentElement.setAttribute('data-power-mode', powerSaveMode);
      };
      
      battery.addEventListener('chargingchange', updatePowerMode);
      battery.addEventListener('levelchange', updatePowerMode);
      updatePowerMode();
    });
  }
}

/**
 * Error boundary fallback component
 */
function ErrorFallback({ error }: { error: Error }): JSX.Element {
  return (
    <div role="alert" style={{ padding: '20px' }}>
      <h2>Something went wrong</h2>
      <pre style={{ color: 'red' }}>{error.message}</pre>
    </div>
  );
}

/**
 * Initializes and renders the React application with strict mode
 * and performance optimizations
 */
async function renderApp(): Promise<void> {
  const rootElement = document.getElementById('root');
  if (!rootElement) {
    throw new Error('Root element not found');
  }

  // Initialize HDR support
  await initializeHDR();

  // Setup performance monitoring
  setupPerformanceMonitoring();

  // Configure root element for GPU acceleration
  rootElement.style.transform = 'translateZ(0)';
  rootElement.style.backfaceVisibility = 'hidden';
  rootElement.style.perspective = '1000px';
  rootElement.style.willChange = 'transform';

  // Create React root with concurrent features
  const root = ReactDOM.createRoot(rootElement);

  // Render application with error boundary and strict mode
  root.render(
    <React.StrictMode>
      <ErrorBoundary FallbackComponent={ErrorFallback}>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
}

// Initialize application
renderApp().catch(error => {
  console.error('Application initialization failed:', error);
});

// Report performance metrics
const reportWebVitals = (onPerfEntry?: (metric: any) => void): void => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    onCLS(onPerfEntry);
    onFID(onPerfEntry);
    onLCP(onPerfEntry);
    onFCP(onPerfEntry);
    onTTFB(onPerfEntry);
  }
};

export default reportWebVitals;