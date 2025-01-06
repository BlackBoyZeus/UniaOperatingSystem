import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { ThemeProvider } from '@emotion/react';
import { lightTheme, darkTheme, highContrastTheme, ThemeConfig } from '../config/theme.config';

// Constants for theme management
const THEME_STORAGE_KEY = 'tald-theme-mode';
const DEFAULT_THEME_MODE = 'light';
const HDR_CONTRAST_LEVELS = [1, 1.5, 2, 2.5, 3];
const COLOR_SPACE_CHECK_INTERVAL = 5000;

// Theme context type definition
interface ThemeContextType {
  currentTheme: ThemeConfig;
  themeMode: string;
  setThemeMode: (mode: string) => void;
  hdrCapable: boolean;
  p3ColorSpace: boolean;
  contrastLevel: number;
  setContrastLevel: (level: number) => void;
  colorDepth: number;
}

// Create theme context
export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

/**
 * Theme context provider component
 * @version 1.0.0
 */
export const ThemeContextProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
  // Theme state management
  const [themeMode, setThemeMode] = useState<string>(() => {
    const storedMode = localStorage.getItem(THEME_STORAGE_KEY);
    return storedMode || DEFAULT_THEME_MODE;
  });

  // Display capabilities state
  const [hdrCapable, setHdrCapable] = useState<boolean>(false);
  const [p3ColorSpace, setP3ColorSpace] = useState<boolean>(false);
  const [contrastLevel, setContrastLevel] = useState<number>(1);
  const [colorDepth, setColorDepth] = useState<number>(8);

  /**
   * Detect HDR and color space capabilities
   */
  const detectDisplayCapabilities = useCallback(() => {
    // Check HDR support
    const hdrSupport = window.matchMedia('(dynamic-range: high)').matches;
    setHdrCapable(hdrSupport);

    // Check P3 color space support
    const p3Support = CSS.supports('color', 'color(display-p3 0 0 0)');
    setP3ColorSpace(p3Support);

    // Detect color depth
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
      const ext = gl.getExtension('EXT_color_buffer_float');
      setColorDepth(ext ? 10 : 8);
    }
  }, []);

  /**
   * Select appropriate theme based on mode and capabilities
   */
  const selectTheme = useCallback((): ThemeConfig => {
    const themeOptions = {
      colorDepth: {
        bits: colorDepth,
        gamut: p3ColorSpace ? 'display-p3' : 'srgb',
        hdrEnabled: hdrCapable,
      },
    };

    switch (themeMode) {
      case 'dark':
        return darkTheme(themeOptions);
      case 'highContrast':
        return highContrastTheme(themeOptions);
      default:
        return lightTheme(themeOptions);
    }
  }, [themeMode, hdrCapable, p3ColorSpace, colorDepth]);

  /**
   * Handle theme mode changes
   */
  const handleThemeModeChange = useCallback((mode: string) => {
    setThemeMode(mode);
    localStorage.setItem(THEME_STORAGE_KEY, mode);
  }, []);

  /**
   * Handle contrast level changes
   */
  const handleContrastChange = useCallback((level: number) => {
    if (HDR_CONTRAST_LEVELS.includes(level)) {
      setContrastLevel(level);
    }
  }, []);

  // Initialize display capabilities detection
  useEffect(() => {
    detectDisplayCapabilities();

    // Monitor for display capability changes
    const mediaQuery = window.matchMedia('(dynamic-range: high)');
    const handleHdrChange = () => detectDisplayCapabilities();
    mediaQuery.addEventListener('change', handleHdrChange);

    // Periodically check for P3 color space support
    const colorSpaceInterval = setInterval(detectDisplayCapabilities, COLOR_SPACE_CHECK_INTERVAL);

    return () => {
      mediaQuery.removeEventListener('change', handleHdrChange);
      clearInterval(colorSpaceInterval);
    };
  }, [detectDisplayCapabilities]);

  // Create context value
  const contextValue: ThemeContextType = {
    currentTheme: selectTheme(),
    themeMode,
    setThemeMode: handleThemeModeChange,
    hdrCapable,
    p3ColorSpace,
    contrastLevel,
    setContrastLevel: handleContrastChange,
    colorDepth,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <ThemeProvider theme={contextValue.currentTheme}>
        {children}
      </ThemeProvider>
    </ThemeContext.Provider>
  );
};

/**
 * Custom hook for accessing theme context
 * @returns ThemeContextType
 * @throws Error if used outside ThemeContextProvider
 */
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeContextProvider');
  }
  return context;
};