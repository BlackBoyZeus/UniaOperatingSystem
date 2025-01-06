import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { Button, ButtonProps } from '../../../src/components/common/Button';
import { measurePerformance, PerformanceObserver } from 'jest-performance';
import HapticFeedback from '@react-native-community/haptic-feedback';

// Mock the navigator.vibrate API
Object.defineProperty(window.navigator, 'vibrate', {
  value: jest.fn(),
  writable: true,
  configurable: true
});

// Mock the HDR detection API
Object.defineProperty(window, 'matchMedia', {
  value: jest.fn().mockImplementation(query => ({
    matches: query === '(dynamic-range: high)',
    addListener: jest.fn(),
    removeListener: jest.fn()
  }))
});

// Mock power mode detection
const mockPowerMode = {
  matches: false,
  addEventListener: jest.fn(),
  removeEventListener: jest.fn()
};

describe('Button Component', () => {
  let performanceObserver: PerformanceObserver;
  
  beforeEach(() => {
    jest.useFakeTimers();
    performanceObserver = new PerformanceObserver();
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: false,
      addListener: jest.fn(),
      removeListener: jest.fn()
    }));
  });

  afterEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
    performanceObserver.disconnect();
  });

  // Core functionality tests
  it('renders with default props', () => {
    render(<Button>Click me</Button>);
    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
    expect(button).toHaveTextContent('Click me');
    expect(button).toHaveAttribute('type', 'button');
  });

  it('handles different variants correctly', () => {
    const { rerender } = render(<Button variant="primary">Primary</Button>);
    expect(screen.getByRole('button')).toHaveStyle({
      background: expect.stringContaining('var(--color-primary')
    });

    rerender(<Button variant="secondary">Secondary</Button>);
    expect(screen.getByRole('button')).toHaveStyle({
      background: expect.stringContaining('var(--color-secondary')
    });
  });

  it('handles disabled state', () => {
    const handleClick = jest.fn();
    render(<Button disabled onClick={handleClick}>Disabled</Button>);
    const button = screen.getByRole('button');
    
    fireEvent.click(button);
    expect(handleClick).not.toHaveBeenCalled();
    expect(button).toHaveStyle({ opacity: '0.5' });
  });

  // HDR support tests
  it('renders in HDR mode when supported', () => {
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(dynamic-range: high)',
      addListener: jest.fn(),
      removeListener: jest.fn()
    }));

    render(<Button hdrMode="enabled">HDR Button</Button>);
    const button = screen.getByRole('button');
    
    expect(button).toHaveStyle({
      background: expect.stringContaining('-hdr')
    });
  });

  it('applies HDR glow effects on hover', async () => {
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(dynamic-range: high)',
      addListener: jest.fn(),
      removeListener: jest.fn()
    }));

    render(<Button hdrMode="enabled">HDR Button</Button>);
    const button = screen.getByRole('button');
    
    await userEvent.hover(button);
    expect(button).toHaveStyle({
      boxShadow: expect.stringContaining('var(--effect-glow)')
    });
  });

  // Performance tests
  it('maintains input latency below 16ms', async () => {
    const { result } = await measurePerformance(() => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Performance Test</Button>);
      fireEvent.click(screen.getByRole('button'));
    });

    expect(result.duration).toBeLessThan(16);
  });

  it('optimizes GPU acceleration', () => {
    render(<Button>GPU Test</Button>);
    const button = screen.getByRole('button');
    
    expect(button).toHaveStyle({
      transform: 'translateZ(0)',
      willChange: 'transform, background-color'
    });
  });

  // Power optimization tests
  it('adapts to power-save mode', () => {
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(prefers-reduced-motion: reduce)',
      addListener: jest.fn(),
      removeListener: jest.fn()
    }));

    render(<Button powerSaveAware>Power Save</Button>);
    const button = screen.getByRole('button');
    
    expect(button).toHaveStyle({
      transition: expect.stringContaining('var(--animation-duration-power-save)')
    });
  });

  it('disables animations in power-save mode', async () => {
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(prefers-reduced-motion: reduce)',
      addListener: jest.fn(),
      removeListener: jest.fn()
    }));

    render(<Button powerSaveAware>Power Save</Button>);
    const button = screen.getByRole('button');
    
    await userEvent.hover(button);
    expect(button).toHaveStyle({ transform: 'none' });
  });

  // Haptic feedback tests
  it('provides haptic feedback on click', () => {
    const vibrateMock = jest.spyOn(navigator, 'vibrate');
    render(<Button enableHaptic>Haptic</Button>);
    
    fireEvent.click(screen.getByRole('button'));
    expect(vibrateMock).toHaveBeenCalledWith(50);
  });

  // Accessibility tests
  it('meets accessibility requirements', () => {
    render(<Button aria-label="Accessible Button">A11y</Button>);
    const button = screen.getByRole('button');
    
    expect(button).toHaveAttribute('aria-label');
    expect(button).toHaveStyle({
      padding: expect.stringContaining('var(--spacing-unit)'),
      userSelect: 'none',
      touchAction: 'manipulation'
    });
  });

  // Memory leak prevention tests
  it('cleans up event listeners on unmount', () => {
    const { unmount } = render(<Button>Cleanup Test</Button>);
    unmount();
    // Verify no memory leaks using the performance observer
    expect(performanceObserver.getEntries()).toHaveLength(0);
  });

  // Debounce behavior tests
  it('debounces click events', () => {
    jest.useFakeTimers();
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Debounce Test</Button>);
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    fireEvent.click(button);
    
    jest.runAllTimers();
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});