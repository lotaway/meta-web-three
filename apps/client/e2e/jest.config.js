module.exports = {
  rootDir: '..',
  testMatch: ['<rootDir>/e2e/**/*.spec.ts'],
  testTimeout: 120000,
  maxWorkers: 1,
  globalSetup: 'detox/runners/jest/globalSetup',
  globalTeardown: 'detox/runners/jest/globalTeardown',
  testEnvironment: 'detox/runners/jest/testEnvironment',
  reporters: ['detox/runners/jest/reporter', 'jest-junit'],
  setupFilesAfterEnv: ['<rootDir>/e2e/setup.ts'],
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', { configFile: require('path').join(__dirname, 'babel.config.js') }],
  },
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|expo.*|@expo.*|@unimodules|unimodules|react-navigation|@react-navigation.*)/)',
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  verbose: true,
}
