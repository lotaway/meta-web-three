# Client e2e (Detox)

Real-device/emulator UI tests for the React Native client. The suite uses
Detox 20 with `testID`-based page objects (`e2e/pages/index.ts`).

## Prerequisites

- Android SDK + an AVD (default `Pixel_7_API_34`, or set `DETOX_AVD_NAME`).
- A reachable backend for the app's `EXPO_PUBLIC_BACK_API_HOST` (default port
  `10081`). In CI/local runs without a real backend, use the bundled mock
  gateway so the app settles to idle — otherwise Detox's synchronization
  (`isReady`) times out on the app's pending API calls.
- Metro running (`npx expo start --port 8081`) and `adb reverse tcp:8081 tcp:8081`.

## Running (attached to an already-running emulator)

```bash
# 1. Start the mock gateway (host must match the app's configured API host).
HOST=192.168.1.110 PORT=10081 node e2e/mock-gateway.js   # or: yarn e2e:mock-gateway

# 2. Start Metro and port-forward.
npx expo start --port 8081 &
adb reverse tcp:8081 tcp:8081

# 3. Build & install both APKs (debug app + androidTest instrumentation).
yarn e2e:build   # or: cd android && ./gradlew assembleDebug assembleDebugAndroidTest

# 4. Run the suite.
E2E_SKIP=0 npx detox test --config-path e2e/detox.config.js -c android.attached.debug
```

## Why a mock gateway is required

Detox waits for the app to become idle before running assertions (`isReady`).
The client fires several API requests on startup (`HomeContainer`); with no
backend those requests stay pending/retrying, the app never goes idle, and
Detox times out. The mock gateway returns empty `ApiResponse` payloads so the
requests complete quickly.

## Instrumentation & Android build notes (Windows)

- `androidTest` instrumentation lives in
  `app/src/androidTest/java/com/metawebthree/app/DetoxTest.java`; the app
  `build.gradle` declares `testInstrumentationRunner` + the `:detox` module.
- Payment modules need `compileSdk 36`; `packagingOptions.pickFirst libfbjni.so`
  avoids duplicate native-lib merge failures.
- On Windows, build with `-PreactNativeArchitectures=x86_64` to avoid CMake's
  260-character path limit on `armeabi-v7a` codegen objects.

## Visual regression (device matrix)

`e2e/visual.spec.ts` captures the home screen via `device.takeScreenshot()` and
compares it against `e2e/visual-baselines/home.png` using pure `pngjs`
(`e2e/visual-compare.ts`), with a 0.2% pixel-diff threshold (industry standard,
top 100px status bar ignored). This is the RN counterpart of the admin
Playwright visual gate.

```bash
# Regenerate baseline after an approved UI change:
yarn e2e:visual:update
# Compare against baseline:
yarn e2e:visual
```

Evidence (2026-11-22, emulator-5554, mock gateway):
- compare run: `visual diff home: 0.000%` → PASS (1 passed)
- deliberate failure: 10000 pixels corrupted → `0.403%` → FAIL
  (`Expected <= 0.002`), baseline restored → PASS again.

## Evidence - smoke (2026-11-22)

`E2E_SKIP=0 npx detox test -c android.attached.debug e2e/smoke.spec.ts` on
`emulator-5554` with the mock gateway:

```
The app seems to be idle
app smoke: launches and renders the home screen [OK]
PASS e2e/smoke.spec.ts (19.9 s) - 1 passed
```
