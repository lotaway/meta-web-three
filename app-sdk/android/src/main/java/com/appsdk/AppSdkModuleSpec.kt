package com.appsdk

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.turbomodule.core.interfaces.TurboModule

interface AppSdkModuleSpec : TurboModule {
    fun scan()
}

abstract class NativeAppSdkSpec(reactContext: ReactApplicationContext) :
    com.facebook.react.bridge.BaseJavaModule(reactContext), AppSdkModuleSpec
