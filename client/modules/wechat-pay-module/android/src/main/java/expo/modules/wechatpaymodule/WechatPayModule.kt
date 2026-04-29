package expo.modules.wechatpaymodule

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise

class WechatPayModule : Module() {

    override fun definition() = ModuleDefinition {
        Name("WechatPayModule")

        Events("onPaymentResult", "onFinalConfirm")

        AsyncFunction("isWechatInstalled") { ->
            // TODO: 检查是否安装微信
            false
        }

        Function("init") { (appId: String) ->
            // TODO: 初始化微信支付
            println("WechatPayModule init with appId: $appId")
        }

        AsyncFunction("pay") { (params: Map<String, Any>) ->
            // TODO: 实现支付逻辑
            println("WechatPayModule pay: $params")
        }

        Function("emitFinalConfirmEvent") { (message: String) ->
            sendEvent("onFinalConfirm", mapOf(
                "message" to message,
                "timestamp" to System.currentTimeMillis()
            ))
        }
    }
}
