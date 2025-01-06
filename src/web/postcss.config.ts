import type { Config } from 'postcss'
import tailwindcss from 'tailwindcss'
import autoprefixer from 'autoprefixer'
import postcssPresetEnv from 'postcss-preset-env'
import cssnano from 'cssnano'
import tailwindConfig from './tailwind.config'

// Version comments for external dependencies
// postcss: ^8.4.0
// tailwindcss: ^3.3.0
// autoprefixer: ^10.4.0
// postcss-preset-env: ^8.0.0
// cssnano: ^6.0.0

export default {
  plugins: [
    // Tailwind CSS with gaming optimizations
    tailwindcss({
      config: tailwindConfig,
      // Enable gaming-specific optimizations
      future: {
        hoverOnlyWhenSupported: true,
        respectDefaultRingColorOpacity: true,
        disableColorOpacityUtilitiesByDefault: true,
      }
    }),

    // Modern CSS features with HDR and P3 color gamut support
    postcssPresetEnv({
      stage: 3,
      features: {
        'custom-properties': true,
        'nesting-rules': true,
        'color-function': {
          preserve: true,
          enableHDR: true,
          colorSpace: 'display-p3'
        },
        'custom-media-queries': {
          preserve: true,
          appendExtensions: true
        },
        'dynamic-viewport-units': true,
        'container-queries': true,
        'media-query-ranges': true,
        'cascade-layers': true,
        'logical-properties-and-values': true,
        'prefers-color-scheme': true,
        'prefers-reduced-motion': true,
        'dynamic-range': true
      },
      browsers: [
        'last 2 versions',
        'not dead',
        'not ie 11'
      ],
      autoprefixer: {
        grid: true,
        flexbox: 'no-2009'
      },
      preserve: true
    }),

    // Vendor prefix automation for cross-platform compatibility
    autoprefixer({
      grid: true,
      flexbox: 'no-2009',
      supports: true,
      overrideBrowserslist: [
        'last 2 versions',
        'not dead',
        'not ie 11'
      ]
    }),

    // Gaming-optimized CSS minification with HDR preservation
    cssnano({
      preset: [
        'advanced',
        {
          discardComments: {
            removeAll: true
          },
          colormin: {
            preserve: true,
            preserveHDR: true
          },
          reduceIdents: false,
          reduceTransforms: false,
          minifySelectors: {
            optimizeForSpeed: true
          },
          normalizeWhitespace: true,
          mergeLonghand: true,
          mergeRules: true,
          minifyParams: true,
          minifyFontValues: true,
          discardUnused: {
            fontFace: false,
            keyframes: false
          },
          zindex: false,
          uniqueSelectors: true,
          calc: true,
          orderedValues: true
        }
      ]
    })
  ],
  
  // Source map configuration for development
  sourceMap: process.env.NODE_ENV !== 'production',
  
  // Parser options
  parser: 'postcss-scss',
  
  // Map annotation options
  map: {
    inline: false,
    annotation: true,
    sourcesContent: true
  }
} satisfies Config