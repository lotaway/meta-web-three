package com.myble

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.turbomodule.core.interfaces.TurboModule

interface MyBleModuleSpec : TurboModule {
    fun scan()
}

abstract class NativeMyBleSpec(reactContext: ReactApplicationContext) :
    com.facebook.react.bridge.BaseJavaModule(reactContext), MyBleModuleSpec
