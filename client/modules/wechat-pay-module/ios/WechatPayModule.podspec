Pod::Spec.new do |s|
  s.name           = 'wechat-pay-module'
  s.version        = '1.0.0'
  s.summary        = 'Wechat Pay Expo Module'
  s.description    = 'Expo module for Wechat Pay integration'
  s.author         = ''
  s.homepage       = 'https://github.com/lotaway/meta-web-three'
  s.platforms      = { :ios => '15.1', :tvos => '15.1' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  # Swift/Objective-C compatibility
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  s.source_files = 'ios/**/*.{h,m,mm,swift,cpp,c,h}'
  s.resource_bundles = {'wechat_pay_module_privacy' => ['ios/PrivacyInfo.xcprivacy']}
end
