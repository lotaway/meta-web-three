const {
  withInfoPlist,
  withAndroidManifest,
  withDangerousMod
} = require('@expo/config-plugins')
const fs = require('fs')
const path = require('path')

module.exports = (config, { appId }) => {
  config = withInfoPlist(config, c => {
    c.modResults.CFBundleURLTypes = c.modResults.CFBundleURLTypes || []
    c.modResults.CFBundleURLTypes.push({
      CFBundleURLSchemes: [`wx${appId}`]
    })
    return c
  })

  config = withAndroidManifest(config, c => {
    const app = c.modResults.manifest.application[0]
    app.activity = app.activity || []
    app.activity.push({
      $: {
        'android:name': '.wxapi.WXPayEntryActivity',
        'android:exported': 'true'
      }
    })
    return c
  })

  config = withDangerousMod(config, ['android', async c => {
    const pkg = c.android.package
    const dir = path.join(
      c.modRequest.platformProjectRoot,
      'app/src/main/java',
      pkg.replace(/\./g, '/'),
      'wxapi'
    )

    fs.mkdirSync(dir, { recursive: true })

    fs.writeFileSync(
      path.join(dir, 'WXPayEntryActivity.java'),
      `package ${pkg}.wxapi;

import android.app.Activity;
import android.os.Bundle;
import com.tencent.mm.opensdk.modelbase.WXPayEntry;
import com.tencent.mm.opensdk.openapi.IWXAPI;
import com.tencent.mm.opensdk.openapi.WXAPIFactory;

public class WXPayEntryActivity extends Activity implements WXPayEntry {
    private IWXAPI api;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        api = WXAPIFactory.createWXAPI(this, getPackageName());
        api.handleIntent(getIntent(), this);
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        api.handleIntent(intent, this);
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    public void onReq(com.tencent.mm.opensdk.modelbase.BaseReq req) {
    }

    public void onPayResult(int errCode) {
        setResult(errCode);
        finish();
    }
}`
    )

    return c
  }])

  return config
}