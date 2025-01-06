import { Theme, ThemeOptions, createTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';

/**
 * Extended theme configuration interface with HDR and P3 color gamut support
 * @version 1.0.0
 */
export interface ThemeConfig extends Theme {
  palette: {
    primary: {
      main: string;
      p3: string;
      hdr: boolean;
      contrastThreshold: number;
    };
    secondary: {
      main: string;
      p3: string;
      hdr: boolean;
      contrastThreshold: number;
    };
    background: {
      default: string;
      paper: string;
      hdr: boolean;
    };
    text: {
      primary: string;
      secondary: string;
      contrast: number;
    };
  };
  typography: {
    fontFamily: string;
    fontSize: number;
    fontWeightLight: number;
    fontWeightRegular: number;
    fontWeightBold: number;
    gaming: {
      family: string;
      weights: {
        light: number;
        regular: number;
        bold: number;
      };
    };
  };
  colorDepth: {
    bits: number;
    gamut: string;
    hdrEnabled: boolean;
  };
}

// Base theme configuration
const BASE_THEME_OPTIONS: ThemeOptions = {
  typography: {
    fontFamily: "'gaming-font', 'Roboto', 'Arial', sans-serif",
    fontSize: 16,
    fontWeightLight: 300,
    fontWeightRegular: 400,
    fontWeightBold: 700,
  },
  spacing: 8,
  shape: {
    borderRadius: 4,
  },
};

// HDR color adaptations using P3 color space
const HDR_COLOR_ADAPTATIONS = {
  primary: {
    main: 'color(display-p3 0.486 0.302 1)',
    light: 'color(display-p3 0.6 0.4 1)',
    dark: 'color(display-p3 0.3 0.2 0.8)',
  },
  secondary: {
    main: 'color(display-p3 0 0.898 1)',
    light: 'color(display-p3 0.2 0.95 1)',
    dark: 'color(display-p3 0 0.7 0.8)',
  },
};

// 10-bit color depth configuration
const COLOR_DEPTH_CONFIG = {
  tenBit: {
    steps: 1024,
    precision: '10-bit',
    dithering: 'error-diffusion',
  },
};

/**
 * Creates a gaming theme with HDR and dynamic contrast adaptation support
 * @param options - Theme customization options
 * @param colorDepth - Color depth and gamut configuration
 * @returns ThemeConfig - Complete theme configuration
 */
export const createGameTheme = (
  options: ThemeOptions,
  colorDepth: {
    bits: number;
    gamut: string;
    hdrEnabled: boolean;
  }
): ThemeConfig => {
  // Check for HDR support
  const supportsHDR = window.matchMedia('(dynamic-range: high)').matches;
  
  // Check for P3 color space support
  const supportsP3 = CSS.supports('color', 'color(display-p3 0 0 0)');
  
  // Configure color adaptations based on system capabilities
  const colorAdaptations = {
    primary: {
      main: supportsP3 ? HDR_COLOR_ADAPTATIONS.primary.main : '#7B4DFF',
      p3: HDR_COLOR_ADAPTATIONS.primary.main,
      hdr: supportsHDR,
      contrastThreshold: 4.5,
    },
    secondary: {
      main: supportsP3 ? HDR_COLOR_ADAPTATIONS.secondary.main : '#00E5FF',
      p3: HDR_COLOR_ADAPTATIONS.secondary.main,
      hdr: supportsHDR,
      contrastThreshold: 4.5,
    },
    background: {
      default: supportsP3 ? 'color(display-p3 0.1 0.1 0.12)' : '#1A1A1F',
      paper: supportsP3 ? 'color(display-p3 0.15 0.15 0.17)' : '#26262B',
      hdr: supportsHDR,
    },
    text: {
      primary: supportsP3 ? 'color(display-p3 1 1 1)' : '#FFFFFF',
      secondary: supportsP3 ? 'color(display-p3 0.7 0.7 0.7)' : '#B3B3B3',
      contrast: supportsHDR ? 1000 : 21,
    },
  };

  // Create base theme with gaming-specific typography
  const baseTheme = createTheme({
    ...BASE_THEME_OPTIONS,
    ...options,
    palette: {
      mode: 'dark',
      ...colorAdaptations,
    },
    typography: {
      ...BASE_THEME_OPTIONS.typography,
      gaming: {
        family: 'gaming-font',
        weights: {
          light: 300,
          regular: 400,
          bold: 700,
        },
      },
    },
  });

  // Extend theme with HDR and color depth configurations
  return {
    ...baseTheme,
    colorDepth: {
      bits: colorDepth.bits || 10,
      gamut: colorDepth.gamut || 'display-p3',
      hdrEnabled: supportsHDR && colorDepth.hdrEnabled,
    },
  } as ThemeConfig;
};

export default createGameTheme;