Pod::Spec.new do |s|
  s.name           = 'WechatPayModule'
  s.version        = '1.0.0'
  s.summary        = 'Wechat Pay Expo Module'
  s.description    = 'Expo module for Wechat Pay integration'
  s.author         = ''
  s.homepage       = 'https://github.com/lotaway/meta-web-three'
  s.platforms      = { :ios => '15.1', :tvos => '15.1' }
  s.source         = { git: '' }
  s.static_framework = false

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_OBJC_BRIDGING_HEADER' => '$(PODS_TARGET_SRCROOT)/ios/WechatPayModule-Bridging-Header.h',
  }
  s.swift_version = '5.0'

  s.source_files = 'ios/**/*.{h,m,mm,swift,cpp,c,h}'
  s.resource_bundles = {'wechat_pay_module_privacy' => ['ios/PrivacyInfo.xcprivacy']}
end
