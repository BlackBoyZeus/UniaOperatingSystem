import type { JestConfigWithTsJest } from 'ts-jest';

// Jest configuration for TALD UNIA backend testing
// @ts-jest/29.1.0 - TypeScript preprocessor
// @jest/29.0.0 - Testing framework
// @types/jest/29.0.0 - TypeScript definitions
const jestConfig: JestConfigWithTsJest = {
  // Use ts-jest preset for TypeScript support
  preset: 'ts-jest',
  
  // Set Node.js as test environment
  testEnvironment: 'node',
  
  // Define test root directories
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  
  // Supported file extensions
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  
  // Module path aliases matching tsconfig.json
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@lidar/(.*)$': '<rootDir>/src/lidar/$1',
    '^@fleet/(.*)$': '<rootDir>/src/fleet/$1',
    '^@game/(.*)$': '<rootDir>/src/game/$1'
  },
  
  // Enable coverage collection
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'clover', 'json-summary'],
  
  // Coverage thresholds per directory
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    },
    './src/lidar/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    },
    './src/fleet/': {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    }
  },
  
  // Test file patterns
  testMatch: [
    '**/__tests__/**/*.[jt]s?(x)',
    '**/?(*.)+(spec|test).[jt]s?(x)',
    '**/?(*.)+(perf|benchmark).[jt]s?(x)'
  ],
  
  // Paths to ignore
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  
  // Test setup file
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  
  // TypeScript transformation configuration
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      diagnostics: {
        warnOnly: false
      }
    }]
  },
  
  // Global ts-jest configuration
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
      isolatedModules: true
    }
  },
  
  // Enable verbose output
  verbose: true,
  
  // Detect open handles (e.g., unfinished async operations)
  detectOpenHandles: true,
  
  // Force exit after tests complete
  forceExit: true,
  
  // Limit worker threads to 50% of available CPUs
  maxWorkers: '50%',
  
  // Test timeout in milliseconds
  testTimeout: 10000,
  
  // Configure test reporters
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'reports',
      outputName: 'jest-junit.xml',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}',
      ancestorSeparator: ' › ',
      usePathForSuiteName: true
    }]
  ]
};

export default jestConfig;