import ExpoModulesCore

public class WechatPayModule: Module {
  private var appId: String = ""

  public func definition() -> ModuleDefinition {
    Name("WechatPayModule")

    Events("onFinalConfirm")

    Function("init") { (appId: String) in
      self.appId = appId
    }

    AsyncFunction("isWechatInstalled") { () -> Bool in
      return false
    }

    AsyncFunction("pay") { (params: [String: Any]) in
      NSLog("WechatPayModule pay: %@", params)
    }

    Function("emitFinalConfirmEvent") { (message: String) in
      self.sendEvent("onFinalConfirm", ["message": message])
    }
  }
}
