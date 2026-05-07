{
  "name": "meta-web-three-e2e",
  "testDir": "./e2e",
  "testMatch": "**/*.spec.ts",
  "reporters": ["detox-runner-reporter", "jest-junit"],
  "setupFilesAfterEnv": ["./e2e/setup.ts"],
  "transform": {
    "^.+\\.(js|jsx|ts|tsx)$": "babel-jest"
  },
  "transformIgnorePatterns": [
    "node_modules/(?!(react-native|@react-native|expo.*|@expo.*|@unimodules|unimodules|react-navigation|@react-navigation.*)/)"
  ],
  "moduleFileExtensions": ["ts", "tsx", "js", "jsx", "json"],
  "testEnvironment": "node",
  "verbose": true
}