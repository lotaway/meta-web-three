package com.myble

import android.util.Log
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule

@ReactModule(name = MyBleModule.NAME)
class MyBleModule(reactContext: ReactApplicationContext) : NativeMyBleSpec(reactContext) {
    override fun scan() {
        Log.d("MyBle", "Start BLE scan")
    }

    companion object {
        const val NAME = "MyBle"
    }
}
