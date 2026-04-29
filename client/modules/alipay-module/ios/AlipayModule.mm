#import <ExpoModulesCore/ExpoModulesCore.h>
#import <AlipaySDK/CoreSDK/ALipaySDK.h>

@interface AlipayModule : EXModuleBase

@property (nonatomic, strong) NSString *appId;

@end

@implementation AlipayModule

EX_METHOD(init, (NSString *)appId {
  self.appId = appId;
})

EX_METHOD(pay, (NSDictionary *)params resolver:(EXPromiseResolveBlock)resolve rejecter:(EXPromiseRejectBlock)reject {
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
})

@end
