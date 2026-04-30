package expo.modules.alipaymodule

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise

class AlipayModule : Module() {

    override fun definition() = ModuleDefinition {
        Name("AlipayModule")

        Function("init") { appId: String ->
            // 支付宝SDK初始化通常在Application中进行
        }

        AsyncFunction("pay") { promise: Promise, params: Map<String, Any> ->
            val orderString = params["orderString"] as? String
            if (orderString.isNullOrEmpty()) {
                promise.reject("INVALID_PARAM", "orderString is required", null)
                return@AsyncFunction
            }

            // 支付宝SDK尚未集成
            // 请从 https://opendocs.alipay.com/open/54/104509 下载SDK
            // 并在 build.gradle 中添加本地依赖
            promise.reject("SDK_NOT_INTEGRATED", "Alipay SDK not integrated. Please download the SDK from official website.", null)
        }
    }
}
