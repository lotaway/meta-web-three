#import <ExpoModulesCore/ExpoModulesCore.h>

@interface WechatPayModule : EXModuleBase

@end

@implementation WechatPayModule

EX_METHOD(init, (NSString *)appId {
  // TODO: 初始化微信支付，需要集成微信 SDK
  NSLog(@"WechatPayModule init with appId: %@", appId);
})

EX_METHOD(isWechatInstalled, () {
  // TODO: 检查是否安装微信
  return @NO;
})

EX_METHOD(pay, (NSDictionary *)params {
  // TODO: 实现支付逻辑
  NSLog(@"WechatPayModule pay: %@", params);
})

EX_METHOD(emitFinalConfirmEvent, (NSString *)message {
  // TODO: 发送确认事件
  [self sendEvent:@"onFinalConfirm" withBody:@{@"message": message ?: @""}];
})

@end
