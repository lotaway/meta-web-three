import ExpoModulesCore

public class AlipayModule: Module {
  private var appId: String = ""
  
  public func definition() -> ModuleDefinition {
    Name("AlipayModule")
    
    AsyncFunction("init") { (appId: String) in
      self.appId = appId
    }
    
    AsyncFunction("pay") { (params: [String: Any]) -> [String: Any] in
      guard let orderString = params["orderString"] as? String else {
        throw NSError(domain: "INVALID_ORDER", code: 400, userInfo: [NSLocalizedDescriptionKey: "orderString is required"])
      }
      
      // Note: Alipay SDK integration requires additional setup
      // This is a placeholder - the actual SDK call would need proper pod integration
      return ["success": false, "error": "Alipay SDK not linked - requires pod 'AlipaySDK-iOS' properly integrated"]
    }
    
    Function("getSdkVersion") { () -> String in
      return "1.0.0"
    }
  }
}