package com.wechatpay

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule
import com.tencent.mm.opensdk.constants.Build
import com.tencent.mm.opensdk.constants.WXAppModuleDataObject
import com.tencent.mm.opensdk.modelpay.PayReq
import com.tencent.mm.opensdk.modelpay.PayResp
import com.tencent.mm.opensdk.openapi.IWXAPI
import com.tencent.mm.opensdk.openapi.WXAPIFactory

@ReactModule(name = WechatPayModule.NAME)
class WechatPayModule(reactContext: ReactApplicationContext) : NativeWechatPaySpec(reactContext) {

  private var api: IWXAPI? = null

  override fun getName(): String = NAME

  override fun init(appId: String) {
    api = WXAPIFactory.createWXAPI(reactContext, appId, true)
    api?.registerApp(appId)
  }

  override fun isWechatInstalled(promise: Promise) {
    promise.resolve(api?.isWXAppInstalled ?: false)
  }

  override fun pay(params: ReadableMap, promise: Promise) {
    val activity = reactContext.currentActivity
    if (activity == null) {
      promise.reject("ACTIVITY_NOT_FOUND", "Cannot find current activity")
      return
    }

    if (api?.isWXAppInstalled != true) {
      promise.reject("NOT_INSTALLED", "Wechat not installed")
      return
    }

    val req = PayReq()
    req.appId = params.getString("appId") ?: ""
    req.partnerId = params.getString("partnerId")
    req.prepayId = params.getString("prepayId")
    req.nonceStr = params.getString("nonceStr")
    req.timeStamp = params.getString("timeStamp")?.toIntOrNull() ?: 0
    req.packageValue = params.getString("packageValue")
    req.sign = params.getString("sign")

    val success = api?.sendReq(activity, req)
    if (success == true) {
      promise.resolve(null)
    } else {
      promise.reject("PAY_FAILED", "Wechat pay failed")
    }
  }

  companion object {
    const val NAME = "WechatPay"
  }
}