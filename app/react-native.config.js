const path = require('path');
const AppSdkPkg = require('../AppSdk/package.json');

module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: true,
    },
  },
  dependencies: {
    [AppSdkPkg.name]: {
      root: path.join(__dirname, '../AppSdk'),
      platforms: {
        // Codegen script incorrectly fails without this
        // So we explicitly specify the platforms with empty object
        ios: {},
        android: {},
      },
    },
  },
};
