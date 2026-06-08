Pod::Spec.new do |s|
  s.name           = 'AlipayModule'
  s.version        = '1.0.0'
  s.summary        = 'Alipay Expo Module'
  s.description    = 'Expo module for Alipay integration'
  s.author         = ''
  s.homepage       = 'https://github.com/lotaway/meta-web-three'
  s.platforms      = { :ios => '15.1', :tvos => '15.1' }
  s.source         = { git: '' }
  s.dependency 'ExpoModulesCore'
  s.dependency 'AlipaySDK-iOS'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }
  s.swift_version = '5.0'

  s.source_files = '**/*.swift'
end
