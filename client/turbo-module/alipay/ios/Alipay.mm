#import "Alipay.h"

#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>

#import <AlipaySDK/CoreSDK/ALipaySDK.h>

#import <memory>

using namespace facebook;

@implementation Alipay {
  NSString *_appId;
}

#pragma mark - TurboModule

- (std::shared_ptr<react::TurboModule>)getTurboModule:
  (const react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<react::NativeAlipaySpecJSI>(params);
}

#pragma mark - AlipaySpec

- (void)init:(NSString *)appId {
  _appId = appId;
}

- (void)pay:(NSDictionary *)params
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  NSString *orderString = params[@"orderString"];
  if (!orderString || orderString.length == 0) {
    reject(@"INVALID_ORDER", @"orderString is required", nil);
    return;
  }

  [[AlipaySDK defaultService] payOrder:orderString fromScheme:@"alipaydemo" callback:^(NSDictionary *resultDict) {
    NSString *resultStatus = resultDict[@"resultStatus"];
    
    if ([resultStatus isEqualToString:@"9000"]) {
      NSString *result = resultDict[@"result"] ?: @"";
      resolve(result);
    } else if ([resultStatus isEqualToString:@"6001"]) {
      reject(@"6001", @"User canceled", nil);
    } else {
      NSString *memo = resultDict[@"memo"] ?: @"Unknown error";
      reject(resultStatus, memo, nil);
    }
  }];
}

@end