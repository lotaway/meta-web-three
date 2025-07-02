#import <React/RCTBridgeModule.h>
#import <React/RCTTurboModule.h>

@interface MyBle : NSObject <RCTBridgeModule, RCTTurboModule>
@end

@implementation MyBle
RCT_EXPORT_MODULE()

- (void)scan {
  NSLog(@"[MyBle] Start BLE scan");
}
@end
