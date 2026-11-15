// Detox 20 configuration. The Detox 19 `composition` / `session` shape and the
// non-ASCII `launchApp.delete` key are gone; apps, devices and configurations
// are now declared separately. Android is the default because this workspace
// builds with `yarn android` on Windows.
const gradle = process.platform === 'win32' ? 'gradlew.bat' : './gradlew'

module.exports = {
  testRunner: {
    args: {
      '$0': 'jest',
      config: 'e2e/jest.config.js',
    },
    jest: {
      setupTimeout: 180000,
    },
  },
  apps: {
    'android.debug': {
      type: 'android.apk',
      binaryPath: 'android/app/build/outputs/apk/debug/app-debug.apk',
      build: `cd android && ${gradle} assembleDebug assembleAndroidTest -DtestBuildType=debug && cd ..`,
    },
  },
  devices: {
    emulator: {
      type: 'android.emulator',
      device: {
        avdName: process.env.DETOX_AVD_NAME || 'Pixel_7_API_34',
      },
    },
    attached: {
      type: 'android.attached',
      device: {
        adbName: process.env.DETOX_ADB_NAME || '.*',
      },
    },
  },
  configurations: {
    'android.emu.debug': {
      device: 'emulator',
      app: 'android.debug',
    },
    'android.attached.debug': {
      device: 'attached',
      app: 'android.debug',
    },
  },
  artifacts: {
    plugins: {
      screenshot: 'failing',
      video: 'none',
      hierarchy: 'disabled',
    },
  },
  behavior: {
    init: {
      reinstallApp: true,
    },
    launchApp: {
      permissions: {
        notifications: 'allow',
        camera: 'allow',
        microphone: 'allow',
      },
    },
  },
}
