#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#import <AuthenticationServices/AuthenticationServices.h>

@interface Appsdk : NSObject <RCTBridgeModule, ASAuthorizationControllerDelegate, ASAuthorizationControllerPresentationContextProviding>
@end