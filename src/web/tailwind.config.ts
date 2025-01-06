import type { Config } from 'tailwindcss'
import plugin from 'tailwindcss/plugin'

// Version comments for external dependencies
// tailwindcss: ^3.3.0
// autoprefixer: ^10.4.0

export default {
  content: [
    './src/**/*.{ts,tsx,css}',
    './components/**/*.{ts,tsx}',
    './game/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        // HDR-aware gaming color system with P3 gamut support
        primary: {
          DEFAULT: 'color(display-p3 0.486 0.302 1)',
          hover: 'color(display-p3 0.608 0.490 1)',
          muted: 'color(display-p3 0.486 0.302 1 / 0.7)'
        },
        secondary: {
          DEFAULT: 'color(display-p3 0 0.898 1)',
          hover: 'color(display-p3 0.2 0.933 1)',
          muted: 'color(display-p3 0 0.898 1 / 0.7)'
        },
        accent: {
          DEFAULT: 'color(display-p3 1 0.5 0)',
          hover: 'color(display-p3 1 0.6 0.2)',
          muted: 'color(display-p3 1 0.5 0 / 0.7)'
        },
        background: {
          DEFAULT: 'color(display-p3 0.071 0.071 0.071)',
          surface: 'color(display-p3 0.118 0.118 0.118)',
          overlay: 'color(display-p3 0 0 0 / 0.5)'
        },
        text: {
          primary: 'color(display-p3 1 1 1)',
          secondary: 'color(display-p3 1 1 1 / 0.7)'
        },
        game: {
          lidar: 'color(display-p3 0.486 0.302 1 / 0.8)',
          mesh: 'color(display-p3 0 0.898 1 / 0.6)',
          fleet: 'color(display-p3 0.486 0.302 1)'
        },
        status: {
          error: 'color(display-p3 1 0.322 0.322)',
          success: 'color(display-p3 0.412 0.941 0.682)',
          warning: 'color(display-p3 1 0.843 0.251)'
        }
      },
      fontFamily: {
        // Custom gaming typography with variable weights
        gaming: ['TALD-Gaming', 'system-ui', '-apple-system', 'sans-serif'],
        system: ['system-ui', '-apple-system', 'sans-serif'],
        mono: ['TALD-Mono', 'monospace']
      },
      spacing: {
        // 8px base unit grid with safe zones
        base: '8px',
        safe: 'env(safe-area-inset-bottom, 0px)',
        game: {
          width: 'calc(100vw - theme(spacing.base) * 4)',
          height: 'calc(100vh - env(safe-area-inset-bottom, 0px))'
        }
      },
      animation: {
        // Power-aware GPU-accelerated animations
        'game-fade': 'var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1)',
        'game-slide': 'var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1)',
        'game-scale': 'var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1)'
      },
      transitionDuration: {
        power: 'var(--animation-duration)'
      },
      transitionTimingFunction: {
        power: 'cubic-bezier(0.4, 0, 0.2, 1)'
      }
    }
  },
  plugins: [
    // Gaming UI components plugin
    plugin(({ addComponents }) => {
      addComponents({
        '.game-container': {
          width: 'theme(spacing.game.width)',
          height: 'theme(spacing.game.height)',
          overflow: 'hidden',
          position: 'relative',
          backgroundColor: 'theme(colors.background.DEFAULT)'
        },
        '.game-overlay': {
          position: 'absolute',
          top: '0',
          left: '0',
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          backgroundColor: 'theme(colors.background.overlay)'
        }
      })
    }),
    // HDR color system plugin
    plugin(({ addBase }) => {
      addBase({
        '@media (dynamic-range: high)': {
          ':root': {
            '--color-primary': 'theme(colors.primary.DEFAULT)',
            '--color-secondary': 'theme(colors.secondary.DEFAULT)',
            '--color-accent': 'theme(colors.accent.DEFAULT)'
          }
        }
      })
    }),
    // Power-aware animation plugin
    plugin(({ addBase }) => {
      addBase({
        '@media (prefers-reduced-motion: reduce)': {
          '*': {
            '--animation-duration': '0.001ms',
            'animation-duration': '0.001ms',
            'transition-duration': '0.001ms',
            'animation-iteration-count': '1',
            transition: 'none'
          }
        }
      })
    }),
    // LiDAR visualization utilities plugin
    plugin(({ addUtilities }) => {
      addUtilities({
        '.lidar-point': {
          width: '2px',
          height: '2px',
          backgroundColor: 'theme(colors.game.lidar)',
          borderRadius: '50%'
        },
        '.lidar-mesh': {
          stroke: 'theme(colors.game.mesh)',
          strokeWidth: '1px',
          fill: 'none'
        }
      })
    }),
    // Fleet status indicator plugin
    plugin(({ addUtilities }) => {
      addUtilities({
        '.fleet-indicator': {
          width: '12px',
          height: '12px',
          borderRadius: '50%',
          backgroundColor: 'theme(colors.game.fleet)',
          boxShadow: '0 0 10px theme(colors.game.fleet)'
        },
        '.fleet-indicator-active': {
          animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
        }
      })
    })
  ]
} satisfies Config