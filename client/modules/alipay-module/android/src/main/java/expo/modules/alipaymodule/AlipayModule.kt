package expo.modules.alipaymodule

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.Promise
import com.alipay.sdk.app.PayTask
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class AlipayModule : Module() {

    private val mainScope = CoroutineScope(Dispatchers.Main)

    override fun definition() = ModuleDefinition {
        Name("AlipayModule")

        Function("init") { (appId: String) ->
            // 支付宝SDK初始化通常在Application中进行
        }

        AsyncFunction("pay") { (params: Map<String, Any>) ->
            val orderString = params["orderString"] as? String
            if (orderString.isNullOrEmpty()) {
                throw Exception("orderString is required")
            }

            val activity = appContext.activityProvider.currentActivity
                ?: throw Exception("Cannot find current activity")

            val payTask = PayTask(activity)
            val result = payTask.payV2(orderString, true)

            val resultStatus = result["resultStatus"]
            return@AsyncFunction when (resultStatus) {
                "9000" -> result["result"] ?: ""
                "6001" -> throw Exception("User canceled")
                else -> throw Exception(result["memo"] ?: "Unknown error")
            }
        }
    }
}
