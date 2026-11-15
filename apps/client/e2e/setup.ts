jest.setTimeout(120000)

// Detox specs require a simulator or a connected device. Keep them enabled by
// default and allow an explicit opt-out on hosts without one. Lifecycle hooks
// are handled by `detox/runners/jest/testEnvironment`, so this file no longer
// depends on the undeclared `detox-adapter-jest` / `detox-matchers` packages.
if (process.env.E2E_SKIP === '1') {
  global.describe = global.describe.skip
}
