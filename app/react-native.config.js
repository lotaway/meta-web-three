const path = require('path');
const AppSdkPkg = require('../AppSdk/package.json');

const getDirname = () => {
  try {
    return __dirname;
  } catch {
    return path.dirname(require.main.filename);
  }
};

module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: true,
    },
  },
  dependencies: {
    [AppSdkPkg.name]: {
      root: path.join(getDirname(), '../AppSdk'),
      platforms: {
        // Codegen script incorrectly fails without this
        // So we explicitly specify the platforms with empty object
        ios: {},
        android: {},
      },
    },
  },
};
