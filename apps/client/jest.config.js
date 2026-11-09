module.exports = {
  preset: 'jest-expo',
  setupFilesAfterEnv: [
    '<rootDir>/jest.setup.js',
    '<rootDir>/jest.setupAfterEnv.js',
  ],
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg|@tanstack/react-query|@testing-library)',
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  collectCoverageFrom: [
    'app/**/*.{ts,tsx}',
    '!app/**/*.d.ts',
    '!app/**/index.ts',
    '!app/**/types.ts',
    '!app/**/*.styles.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50,
    },
  },
  testMatch: [
    '**/__tests__/**/*.test.{ts,tsx}',
    '**/*.test.{ts,tsx}',
    '**/*.spec.{ts,tsx}',
  ],
  moduleNameMapper: {
    // Keep these in sync with compilerOptions.paths in tsconfig.json.
    // The old '^@/(.*)$' -> '<rootDir>/app/$1' rule pointed at a directory
    // that does not contain the lib/ and api/ modules any more.
    '^@/components/(order|profile|product|checkout|refund)/(.*)$': '<rootDir>/src/components/$1/$2',
    '^@/lib/(.*)$': '<rootDir>/src/lib/$1',
    '^@/(.*)$': '<rootDir>/$1',
    '^@components/(.*)$': '<rootDir>/components/$1',
    '^@hooks/(.*)$': '<rootDir>/hooks/$1',
    '^@utils/(.*)$': '<rootDir>/utils/$1',
  },
  // Do not override `testEnvironment` here: jest-expo (via react-native's
  // jest-preset) installs react-native-env.js, which initialises NativeModules
  // and UIManager. Forcing 'node' leaves mockNativeModules.UIManager undefined
  // and makes jest-expo's setup throw
  // "TypeError: Object.defineProperty called on non-object".
  verbose: true,
  collectCoverage: process.env.COVERAGE === 'true',
  coverageReporters: ['text', 'lcov', 'html', 'clover'],
  coverageDirectory: '<rootDir>/coverage',
  testPathIgnorePatterns: [
    '/node_modules/',
    '/e2e/',
    '/android/',
    '/ios/',
  ],
}