#import <AppsdkSpec/AppsdkSpec.h>
#import <AuthenticationServices/AuthenticationServices.h>

@interface Appsdk : NativeAppsdkSpecBase <NativeAppsdkSpec, ASAuthorizationControllerDelegate, ASAuthorizationControllerPresentationContextProviding>
@end