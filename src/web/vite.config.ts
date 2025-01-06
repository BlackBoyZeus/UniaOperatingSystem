import { defineConfig } from 'vite'; // ^4.3.0
import react from '@vitejs/plugin-react'; // ^4.0.0
import tsconfigPaths from 'vite-tsconfig-paths'; // ^4.2.0
import webrtc from 'vite-plugin-webrtc'; // ^1.1.0
import { resolve } from 'path';

// Default export providing comprehensive Vite configuration
export default defineConfig(({ command, mode }) => {
  const isProduction = mode === 'production';

  return {
    plugins: [
      react({
        // Enable Fast Refresh for development
        fastRefresh: !isProduction,
        // Enable JSX runtime optimization
        jsxRuntime: 'automatic',
        // Enable babel optimization plugins in production
        babel: {
          plugins: isProduction ? [
            ['transform-remove-console', { exclude: ['error', 'warn'] }],
            'transform-react-remove-prop-types'
          ] : []
        }
      }),
      tsconfigPaths(),
      webrtc({
        // WebRTC configuration for mesh networking
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ],
        maxPeerConnections: 32, // Maximum fleet size
        timeout: 5000
      })
    ],

    build: {
      target: 'esnext',
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: true,
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: isProduction,
          pure_funcs: isProduction ? ['console.log'] : [],
          passes: 3
        },
        mangle: {
          safari10: true
        }
      },
      rollupOptions: {
        output: {
          manualChunks: {
            // Vendor chunk optimization
            vendor: ['react', 'react-dom'],
            // Feature-specific chunks
            lidar: [/\/@lidar\//],
            fleet: [/\/@fleet\//],
            game: ['three', 'webrtc-adapter'],
            // Core functionality chunks
            core: [/\/@components\//, /\/@services\//],
            utils: [/\/@utils\//, /\/@hooks\//, /\/@contexts\//]
          },
          assetFileNames: 'assets/[name]-[hash][extname]',
          chunkFileNames: '[name]-[hash].js',
          entryFileNames: '[name]-[hash].js'
        }
      },
      // Performance optimizations
      reportCompressedSize: false,
      chunkSizeWarningLimit: 1000
    },

    server: {
      port: 3000,
      host: true,
      strictPort: true,
      cors: true,
      hmr: {
        protocol: 'ws',
        host: 'localhost',
        port: 3000,
        timeout: 5000
      },
      watch: {
        ignored: ['**/node_modules/**', '**/dist/**']
      }
    },

    preview: {
      port: 3000,
      strictPort: true,
      host: true
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
        '@components': resolve(__dirname, 'src/components'),
        '@services': resolve(__dirname, 'src/services'),
        '@utils': resolve(__dirname, 'src/utils'),
        '@hooks': resolve(__dirname, 'src/hooks'),
        '@contexts': resolve(__dirname, 'src/contexts'),
        '@types': resolve(__dirname, 'src/types'),
        '@interfaces': resolve(__dirname, 'src/interfaces'),
        '@assets': resolve(__dirname, 'src/assets'),
        '@styles': resolve(__dirname, 'src/styles'),
        '@lidar': resolve(__dirname, 'src/lidar'),
        '@game': resolve(__dirname, 'src/game'),
        '@fleet': resolve(__dirname, 'src/fleet')
      }
    },

    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'three',
        'webrtc-adapter',
        '@lidar/core',
        '@fleet/manager'
      ],
      exclude: ['@internal/*']
    },

    define: {
      __LIDAR_ENABLED__: 'true',
      __FLEET_SIZE__: '32',
      __BUILD_VERSION__: JSON.stringify(process.env.npm_package_version),
      'process.env.VITE_LIDAR_ENDPOINT': JSON.stringify(process.env.VITE_LIDAR_ENDPOINT),
      'process.env.VITE_FLEET_WS_URL': JSON.stringify(process.env.VITE_FLEET_WS_URL)
    },

    // Performance optimizations for LiDAR and mesh networking
    worker: {
      format: 'es',
      plugins: []
    },

    // Environment-specific configurations
    envPrefix: 'VITE_',
    clearScreen: false,
    logLevel: 'info'
  };
});