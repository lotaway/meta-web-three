#import <Foundation/Foundation.h>
#import <AppsdkSpec/AppsdkSpec.h>
#import <AuthenticationServices/AuthenticationServices.h>

@interface Appsdk : NSObject <
AppsdkSpec,
ASAuthorizationControllerDelegate,
ASAuthorizationControllerPresentationContextProviding
>

@end
