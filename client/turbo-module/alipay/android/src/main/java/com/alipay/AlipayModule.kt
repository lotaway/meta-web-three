package com.alipay

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.module.annotations.ReactModule
import com.alipay.sdk.app.PayTask
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@ReactModule(name = AlipayModule.NAME)
class AlipayModule(reactContext: ReactApplicationContext) : NativeAlipaySpec(reactContext) {

  private val mainScope = CoroutineScope(Dispatchers.Main)

  override fun getName(): String = NAME

  override fun init(appId: String) {
    //支付宝SDK初始化通常在Application中进行
  }

  override fun pay(params: ReadableMap, promise: Promise) {
    val orderString = params.getString("orderString")
    if (orderString.isNullOrEmpty()) {
      promise.reject("INVALID_ORDER", "orderString is required")
      return
    }

    mainScope.launch {
      try {
        val activity = reactContext.currentActivity
        if (activity == null) {
          promise.reject("ACTIVITY_NOT_FOUND", "Cannot find current activity")
          return@launch
        }

        val payTask = PayTask(activity)
        val result = payTask.payV2(orderString, true)

        val resultStatus = result["resultStatus"]
        when (resultStatus) {
          "9000" -> {
            val resultMsg = result["result"] ?: ""
            promise.resolve(resultMsg)
          }
          "6001" -> {
            promise.reject("6001", "User canceled")
          }
          else -> {
            val memo = result["memo"] ?: "Unknown error"
            promise.reject(resultStatus, memo)
          }
        }
      } catch (e: Exception) {
        promise.reject("PAY_ERROR", e.message)
      }
    }
  }

  companion object {
    const val NAME = "Alipay"
  }
}