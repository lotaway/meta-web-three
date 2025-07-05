package com.appsdk

import android.util.Log
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule

@ReactModule(name = AppSdkModule.NAME)
class AppSdkModule(reactContext: ReactApplicationContext) : NativeAppSdkSpec(reactContext) {
    override fun scan() {
        Log.d("AppSdk", "Start BLE scan")
    }

    companion object {
        const val NAME = "AppSdk"
    }
}
