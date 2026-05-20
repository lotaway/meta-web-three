#import "Appsdk.h"

#import <React/RCTUtils.h>
#import <AuthenticationServices/AuthenticationServices.h>

@implementation Appsdk {
  RCTPromiseResolveBlock _resolve;
  RCTPromiseRejectBlock _reject;
}

RCT_EXPORT_MODULE();

#pragma mark - Utils

- (NSData *)randomData:(NSUInteger)length {
  NSMutableData *data = [NSMutableData dataWithLength:length];
  SecRandomCopyBytes(kSecRandomDefault, length, data.mutableBytes);
  return data;
}

- (NSString *)b64url:(NSData *)data {
  NSString *base64 = [data base64EncodedStringWithOptions:0];
  base64 = [base64 stringByReplacingOccurrencesOfString:@"+" withString:@"-"];
  base64 = [base64 stringByReplacingOccurrencesOfString:@"/" withString:@"_"];
  base64 = [base64 stringByReplacingOccurrencesOfString:@"=" withString:@""];
  return base64;
}

- (NSData *)b64urlDecode:(NSString *)str {
  NSString *base64 = [str stringByReplacingOccurrencesOfString:@"-" withString:@"+"];
  base64 = [base64 stringByReplacingOccurrencesOfString:@"_" withString:@"/"];
  NSInteger padding = 4 - (base64.length % 4);
  if (padding < 4) {
    base64 = [base64 stringByPaddingToLength:base64.length + padding withString:@"=" startingAtIndex:0];
  }
  return [[NSData alloc] initWithBase64EncodedString:base64 options:0];
}

- (void)clear {
  _resolve = nil;
  _reject = nil;
}

#pragma mark - Passkey Create

RCT_EXPORT_METHOD(createPasskey:(NSString *)rpId
                  userName:(NSString *)userName
                   resolve:(RCTPromiseResolveBlock)resolve
                    reject:(RCTPromiseRejectBlock)reject)
{
  if (@available(iOS 16.0, *)) {

    if (_resolve) {
      reject(@"BUSY", @"Another request running", nil);
      return;
    }

    _resolve = resolve;
    _reject = reject;

    ASAuthorizationPlatformPublicKeyCredentialProvider *provider =
      [[ASAuthorizationPlatformPublicKeyCredentialProvider alloc] initWithRelyingPartyIdentifier:rpId];

    NSData *challenge = [self randomData:32];
    NSData *userId = [self randomData:16];

    ASAuthorizationPlatformPublicKeyCredentialRegistrationRequest *request =
      [provider createCredentialRegistrationRequestWithChallenge:challenge
                                                          name:userName
                                                        userID:userId];

    request.userVerificationPreference =
      ASAuthorizationPublicKeyCredentialUserVerificationPreferenceRequired;

    ASAuthorizationController *controller =
      [[ASAuthorizationController alloc] initWithAuthorizationRequests:@[request]];

    controller.delegate = self;
    controller.presentationContextProvider = self;

    [controller performRequests];

  } else {
    reject(@"UNSUPPORTED", @"iOS 16+", nil);
  }
}

#pragma mark - Passkey Auth

RCT_EXPORT_METHOD(authenticatePasskey:(NSString *)rpId
                      challenge:(NSString *)challenge
                       resolve:(RCTPromiseResolveBlock)resolve
                        reject:(RCTPromiseRejectBlock)reject)
{
  if (@available(iOS 16.0, *)) {

    if (_resolve) {
      reject(@"BUSY", @"Another request running", nil);
      return;
    }

    NSData *challengeData = [self b64urlDecode:challenge];
    if (!challengeData) {
      reject(@"INVALID_CHALLENGE", @"invalid base64url", nil);
      return;
    }

    _resolve = resolve;
    _reject = reject;

    ASAuthorizationPlatformPublicKeyCredentialProvider *provider =
      [[ASAuthorizationPlatformPublicKeyCredentialProvider alloc] initWithRelyingPartyIdentifier:rpId];

    ASAuthorizationPlatformPublicKeyCredentialAssertionRequest *request =
      [provider createCredentialAssertionRequestWithChallenge:challengeData];

    request.userVerificationPreference =
      ASAuthorizationPublicKeyCredentialUserVerificationPreferenceRequired;

    ASAuthorizationController *controller =
      [[ASAuthorizationController alloc] initWithAuthorizationRequests:@[request]];

    controller.delegate = self;
    controller.presentationContextProvider = self;

    [controller performRequests];

  } else {
    reject(@"UNSUPPORTED", @"iOS 16+", nil);
  }
}

#pragma mark - Delegate

- (void)authorizationController:(ASAuthorizationController *)controller
didCompleteWithAuthorization:(ASAuthorization *)authorization API_AVAILABLE(ios(16.0))
{
  id credential = authorization.credential;

  if ([credential isKindOfClass:[ASAuthorizationPlatformPublicKeyCredentialRegistration class]]) {

    ASAuthorizationPlatformPublicKeyCredentialRegistration *cred = credential;

    NSDictionary *res = @{
      @"id": [self b64url:cred.credentialID],
      @"rawId": [self b64url:cred.credentialID],
      @"type": @"public-key",
      @"response": @{
        @"attestationObject": [self b64url:cred.rawAttestationObject],
        @"clientDataJSON": [self b64url:cred.rawClientDataJSON]
      }
    };

    _resolve(res);
    [self clear];
    return;
  }

  if ([credential isKindOfClass:[ASAuthorizationPlatformPublicKeyCredentialAssertion class]]) {

    ASAuthorizationPlatformPublicKeyCredentialAssertion *cred = credential;

    NSDictionary *res = @{
      @"id": [self b64url:cred.credentialID],
      @"rawId": [self b64url:cred.credentialID],
      @"type": @"public-key",
      @"response": @{
        @"clientDataJSON": [self b64url:cred.rawClientDataJSON],
        @"authenticatorData": [self b64url:cred.rawAuthenticatorData],
        @"signature": [self b64url:cred.signature],
        @"userHandle": cred.userID ? [self b64url:cred.userID] : [NSNull null]
      }
    };

    _resolve(res);
    [self clear];
    return;
  }

  _reject(@"UNKNOWN", @"credential type", nil);
  [self clear];
}

- (void)authorizationController:(ASAuthorizationController *)controller
didCompleteWithError:(NSError *)error API_AVAILABLE(ios(16.0))
{
  if (_reject) {
    _reject(@"AUTH_ERROR", error.localizedDescription, error);
  }
  [self clear];
}

#pragma mark - UI

- (ASPresentationAnchor)presentationAnchorForController:(ASAuthorizationController *)controller {
  UIViewController *vc = RCTPresentedViewController();
  if (vc.view.window) {
    return vc.view.window;
  }
  for (UIWindow *window in UIApplication.sharedApplication.windows) {
    if (window.isKeyWindow) {
      return window;
    }
  }
  return nil;
}

@end