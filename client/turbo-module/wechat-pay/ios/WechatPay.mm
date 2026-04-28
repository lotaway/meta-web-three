#import "WechatPay.h"

#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>

#import <WeChatOpenSDK/WXApi.h>

#import <memory>

using namespace facebook;

@interface WechatPay () <WXApiDelegate>
@end

@implementation WechatPay {
  NSString *_appId;
  RCTPromiseResolveBlock _payResolve;
  RCTPromiseRejectBlock _payReject;
  BOOL _hasListeners;
}

- (std::shared_ptr<react::TurboModule>)getTurboModule:
  (const react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<react::NativeWechatPaySpecJSI>(params);
}

- (NSArray<NSString *> *)supportedEvents {
  return @[@"WechatPayResult", @"WechatPayFinalConfirm"];
}

- (void)startObserving {
  _hasListeners = YES;
}

- (void)stopObserving {
  _hasListeners = NO;
}

- (void)emitFinalConfirmEvent:(NSString *)message {
  if (!_hasListeners) return;

  NSDictionary *event = @{
    @"message": message ?: @"",
    @"timestamp": @([[NSDate date] timeIntervalSince1970] * 1000)
  };

  [self sendEventWithName:@"WechatPayFinalConfirm" body:event];
}

- (void)init:(NSString *)appId {
  _appId = appId;
  [WXApi registerApp:appId universalLink:@"https://your-universal-link.com/"];
  [WXApi handleOpenURL:[NSURL URLWithString:[NSString stringWithFormat:@"wx%@://", appId]] delegate:self];
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

  _payResolve = resolve;
  _payReject = reject;

  PayReq *req = [[PayReq alloc] init];
  req.partnerId = params[@"partnerId"];
  req.prepayId = params[@"prepayId"];
  req.nonceStr = params[@"nonceStr"];
  req.timeStamp = [params[@"timeStamp"] intValue];
  req.package = params[@"packageValue"];
  req.sign = params[@"sign"];
  req.openID = _appId;

  [WXApi sendReq:req completion:nil];
}

- (void)onResp:(BaseResp *)resp {
  if (!_hasListeners) return;
  if (![resp isKindOfClass:[PayResp class]]) return;

  PayResp *payResp = (PayResp *)resp;
  [self handlePayCallback:payResp];
  [self emitPayEvent:payResp];
}

- (void)handlePayCallback:(PayResp *)payResp {
  if (!_payResolve || !_payReject) return;

  switch (payResp.errCode) {
    case WXSuccess:
      _payResolve(@{@"transactionId": payResp.transactionId ?: @""});
      break;
    case WXUserCancelled:
      _payReject(@"USER_CANCEL", @"User cancelled", nil);
      break;
    default:
      _payReject(@"PAY_FAILED", payResp.errStr ?: @"Payment failed", nil);
      break;
  }
  _payResolve = nil;
  _payReject = nil;
}

- (void)emitPayEvent:(PayResp *)payResp {
  NSDictionary *event = @{
    @"errCode": @(payResp.errCode),
    @"errStr": payResp.errStr ?: @"",
    @"transactionId": payResp.transactionId ?: @""
  };
  [self sendEventWithName:@"WechatPayResult" body:event];
}

@end
