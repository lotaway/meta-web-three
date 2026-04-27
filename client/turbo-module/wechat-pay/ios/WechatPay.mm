#import "WechatPay.h"

#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>

#import <WeChatOpenSDK/WXApi.h>

#import <memory>

using namespace facebook;

@implementation WechatPay {
  NSString *_appId;
}

#pragma mark - TurboModule

- (std::shared_ptr<react::TurboModule>)getTurboModule:
  (const react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<react::NativeWechatPaySpecJSI>(params);
}

#pragma mark - WechatPaySpec

- (void)init:(NSString *)appId {
  _appId = appId;
  [WXApi registerApp:appId universalLink:@"https://your-universal-link.com/"];
}

- (void)isWechatInstalled:(RCTPromiseResolveBlock)resolve
                   reject:(RCTPromiseRejectBlock)reject
{
  resolve(@([WXApi isWXAppInstalled]));
}

- (void)pay:(NSDictionary *)params
       resolve:(RCTPromiseResolveBlock)resolve
        reject:(RCTPromiseRejectBlock)reject
{
  if (![WXApi isWXAppInstalled]) {
    reject(@"NOT_INSTALLED", @"Wechat not installed", nil);
    return;
  }

  PayReq *req = [[PayReq alloc] init];
  req.partnerId = params[@"partnerId"];
  req.prepayId = params[@"prepayId"];
  req.nonceStr = params[@"nonceStr"];
  req.timeStamp = [params[@"timeStamp"] intValue];
  req.package = params[@"packageValue"];
  req.sign = params[@"sign"];
  req.openID = _appId;

  [WXApi sendReq:req completion:^(BOOL success) {
    if (success) {
      resolve(nil);
    } else {
      reject(@"PAY_FAILED", @"Wechat pay failed", nil);
    }
  }];
}

@end