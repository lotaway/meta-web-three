package com.appsdk

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.rtncalculator.NativeAppSdkSpec

class AppSdkModule(reactContext: ReactApplicationContext) : NativeAppSdkSpec(reactContext) {

  override fun getName() = NAME

  override fun add(a: Double, b: Double, promise: Promise) {
    promise.resolve(a + b)
  }

  companion object {
    const val NAME = "AppSdk"
  }
}