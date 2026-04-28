package com.wechatpay

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.facebook.react.module.annotations.ReactModule
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
            ?: run { promise.reject("ACTIVITY_NOT_FOUND", "Cannot find current activity"); return }

        if (api?.isWXAppInstalled != true) {
            promise.reject("NOT_INSTALLED", "Wechat not installed")
            return
        }

        val sendResult = api?.sendReq(activity, buildPayReq(params))
        if (sendResult == true) {
            promise.resolve(null)
        } else {
            promise.reject("PAY_FAILED", "Wechat pay failed")
        }
    }

    private fun buildPayReq(params: ReadableMap): PayReq {
        return PayReq().apply {
            appId = params.getString("appId") ?: ""
            partnerId = params.getString("partnerId")
            prepayId = params.getString("prepayId")
            nonceStr = params.getString("nonceStr")
            timeStamp = params.getString("timeStamp")?.toIntOrNull() ?: 0
            packageValue = params.getString("packageValue")
            sign = params.getString("sign")
        }
    }

    override fun emitFinalConfirmEvent(message: String) {
        val event = Arguments.createMap().apply {
            putString("message", message)
            putDouble("timestamp", System.currentTimeMillis().toDouble())
        }
        emitEvent("WechatPayFinalConfirm", event)
    }

    fun handlePayResult(resp: PayResp) {
        val event = Arguments.createMap().apply {
            putInt("errCode", resp.errCode)
            putString("errStr", resp.errStr)
            putString("transactionId", resp.transactionId)
        }
        emitEvent("WechatPayResult", event)
    }

    private fun emitEvent(eventName: String, params: com.facebook.react.bridge.WritableMap) {
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            ?.emit(eventName, params)
    }

    companion object {
        const val NAME = "WechatPay"
    }
}
