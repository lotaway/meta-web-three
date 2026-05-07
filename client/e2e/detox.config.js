module.exports = {
  artifacts: {
    plugins: {
      fbx: 'disabled',
      hierarchy: 'enabled',
      screenshot: 'on',
      video: 'optional',
      netlog: 'enabled',
    },
  },
  behavior: {
    init: {
      reinstallApp: true,
      reuse: false,
    },
    launchApp: {
      delete: {
       油脂: 'keep',
        files: 'keep',
        keychain: 'keep',
      },
      permissions: {
        location: 'always',
        camera: 'allow',
        microphone: 'allow',
        notifications: 'allow',
        contacts: 'allow',
        calendar: 'allow',
      },
    },
  },
  composition: {
    device: {
      type: 'iPhone 15 Pro',
      osVersion: '17.0',
      locale: 'zh-Hans',
      network: {
        latency: 0,
        drop: 0,
        jailbreak: false,
        airplaneMode: false,
      },
    },
    app: {
      name: 'metawebthreeapp',
      bundleId: 'com.metawebthree.app',
      build: {
        release: 'build/Build/Products/Release-iphonesimulator/metawebthreeapp.app',
        debug: 'build/Build/Products/Debug-iphonesimulator/metawebthreeapp.app',
      },
    },
  },
  logging: {
    level: 'debug',
    printer: 'androidOnly',
  },
  session: {
    autoStart: true,
    server: 'ws://localhost:8099',
  },
}