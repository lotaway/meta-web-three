import AlipaySDK
import ExpoModulesCore

public class AlipayModule: Module {
  private var appId: String = ""

  public func definition() -> ModuleDefinition {
    Name("AlipayModule")

    Function("init") { (appId: String) in
      self.appId = appId
    }

    AsyncFunction("pay") { (params: [String: Any], promise: Promise) in
      guard let orderString = params["orderString"] as? String, !orderString.isEmpty else {
        promise.reject("INVALID_ORDER", "orderString is required")
        return
      }

      AlipaySDK.defaultService().payOrder(
        orderString,
        fromScheme: "alipaydemo",
        callback: { resultDict in
          guard let resultDict = resultDict else {
            promise.reject("UNKNOWN", "No response from Alipay")
            return
          }

          let resultStatus = resultDict["resultStatus"] as? String ?? ""
          if resultStatus == "9000" {
            let result = resultDict["result"] as? String ?? ""
            promise.resolve(result)
          } else if resultStatus == "6001" {
            promise.reject("6001", "User canceled")
          } else {
            let memo = resultDict["memo"] as? String ?? "Unknown error"
            promise.reject(resultStatus, memo)
          }
        }
      )
    }
  }
}
