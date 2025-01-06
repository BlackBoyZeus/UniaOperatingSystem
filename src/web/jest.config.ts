import type { Config } from '@jest/types';

const config: Config.InitialOptions = {
  // Use ts-jest as the base preset for TypeScript support
  preset: 'ts-jest',

  // Configure jsdom test environment for web component testing
  testEnvironment: 'jsdom',

  // Setup files to run after environment is setup
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],

  // Module name mapping for path aliases and static assets
  moduleNameMapper: {
    // Source path aliases
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@services/(.*)$': '<rootDir>/src/services/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1',

    // Static asset mocks
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/tests/mocks/fileMock.ts'
  },

  // Transform configuration for TypeScript and JavaScript files
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: '<rootDir>/tsconfig.json',
      diagnostics: true
    }],
    '^.+\\.jsx?$': 'babel-jest'
  },

  // Test file patterns
  testRegex: '(/__tests__/.*|(\\.|/)(test|spec))\\.[jt]sx?$',

  // File extensions to consider
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],

  // Coverage configuration
  coverageDirectory: '<rootDir>/coverage',
  coverageReporters: [
    'text',
    'lcov',
    'json-summary',
    'html'
  ],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/vite-env.d.ts',
    '!src/main.tsx',
    '!src/**/*.stories.{ts,tsx}',
    '!src/**/__mocks__/**'
  ],

  // TypeScript-specific settings
  globals: {
    'ts-jest': {
      tsconfig: '<rootDir>/tsconfig.json',
      diagnostics: true,
      isolatedModules: true
    }
  },

  // Network testing timeout configuration (10 seconds)
  testTimeout: 10000,

  // Mock behavior configuration
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Performance optimization
  maxWorkers: '50%',

  // Verbose output for detailed test reporting
  verbose: true,

  // Test environment options
  testEnvironmentOptions: {
    url: 'http://localhost'
  },

  // Watch plugins for better development experience
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ]
};

export default config;