#import <React/RCTBridgeModule.h>
#import <React/RCTLog.h>
#import <React/RCTTurboModule.h>

@interface AppSdk : NSObject <RCTBridgeModule, RCTTurboModule>
@end

@implementation AppSdk

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(scan) {
  NSLog(@"[AppSdk] Start BLE scan");
}

@end
