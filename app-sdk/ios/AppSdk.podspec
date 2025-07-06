require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name         = "AppSdk"
  s.version      = "1.0.0"
  s.summary      = "A Sdk native module for React Native App"
  s.homepage     = "https://github.com/lotaway"
  s.license      = { :type => "MIT" }
  s.author       = { "Way Luk" => "lotawy@foxmail.com" }
  s.platform     = :ios, "13.0"
  s.source       = { :path => "." }
  s.source_files = "*.{h,m,mm,swift}"

  s.dependency "React-Core"
  # s.dependency "React-RCTBridge"
  s.dependency "React-CoreModules"
  s.dependency "ReactCommon"
  s.dependency "React-Codegen"
  
  s.pod_target_xcconfig = {
    "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/ReactCommon\" \"$(PODS_ROOT)/React-Core/React\" \"$(PODS_ROOT)/React-Core\" \"$(PODS_ROOT)/boost\" \"$(PODS_ROOT)/RCT-Folly\""
  }

  s.compiler_flags = "-DRCT_NEW_ARCH_ENABLED=1"
end