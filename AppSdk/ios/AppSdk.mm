#import "AppSdkSpec.h"
#import "AppSdk.h"

@implementation AppSdk

RCT_REMAP_METHOD(add, addA:(NSInteger)a
                      andB:(NSInteger)b
                withResolver:(RCTPromiseResolveBlock) resolve
                withRejecter:(RCTPromiseRejectBlock) reject)
{
    NSNumber *result = [[NSNumber alloc] initWithInteger:a+b];
    resolve(result);
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeAppSdkSpecJSI>(params);
}

@end