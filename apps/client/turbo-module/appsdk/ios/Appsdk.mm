#import "Appsdk.h"

#import <React/RCTUtils.h>
#import <AuthenticationServices/AuthenticationServices.h>
#import <CommonCrypto/CommonCrypto.h>



@implementation Appsdk {
  RCTPromiseResolveBlock _resolve;
  RCTPromiseRejectBlock _reject;
}

RCT_EXPORT_MODULE();

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

#pragma mark - NativeAppsdkSpec

- (NSString *)generateRequestSignature:(NSDictionary *)params secretKey:(NSString *)secretKey {
  NSMutableArray<NSString *> *paramPairs = [NSMutableArray array];
  NSArray *sortedKeys = [[params allKeys] sortedArrayUsingSelector:@selector(compare:)];
  for (NSString *key in sortedKeys) {
    [paramPairs addObject:[NSString stringWithFormat:@"%@=%@", key, params[key] ?: @""]];
  }
  NSString *message = [[paramPairs componentsJoinedByString:@"&"] stringByAppendingString:secretKey];

  NSData *data = [message dataUsingEncoding:NSUTF8StringEncoding];
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.bytes, (CC_LONG)data.length, digest);

  NSMutableString *hash = [NSMutableString string];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
    [hash appendFormat:@"%02x", digest[i]];
  }
  return [hash copy];
}

- (NSString *)preciseAmountSum:(NSString *)amountA amountB:(NSString *)amountB {
  NSDecimalNumber *a = [NSDecimalNumber decimalNumberWithString:amountA];
  NSDecimalNumber *b = [NSDecimalNumber decimalNumberWithString:amountB];
  return [[a decimalNumberByAdding:b] stringValue];
}

- (NSString *)computeOrderTotal:(NSString *)unitPrice quantity:(double)quantity discountAmount:(NSString *)discountAmount shippingFee:(NSString *)shippingFee {
  NSDecimalNumber *price = [NSDecimalNumber decimalNumberWithString:unitPrice];
  NSDecimalNumber *qty = [[NSDecimalNumber alloc] initWithDouble:quantity];
  NSDecimalNumber *subtotal = [price decimalNumberByMultiplyingBy:qty];
  NSDecimalNumber *discount = [NSDecimalNumber decimalNumberWithString:discountAmount];
  NSDecimalNumber *shipping = [NSDecimalNumber decimalNumberWithString:shippingFee];
  NSDecimalNumber *total = [[subtotal decimalNumberBySubtracting:discount] decimalNumberByAdding:shipping];
  return [total stringValue];
}

- (NSString *)hmacSign:(NSString *)message signingKey:(NSString *)signingKey {
  const char *cKey = [signingKey cStringUsingEncoding:NSUTF8StringEncoding];
  const char *cData = [message cStringUsingEncoding:NSUTF8StringEncoding];

  unsigned char cHMAC[CC_SHA256_DIGEST_LENGTH];
  CCHmac(kCCHmacAlgSHA256, cKey, strlen(cKey), cData, strlen(cData), cHMAC);

  NSMutableString *hash = [NSMutableString string];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
    [hash appendFormat:@"%02x", cHMAC[i]];
  }
  return [hash copy];
}

- (NSString *)createNonce {
  uuid_t uuid;
  uuid_generate(uuid);
  uuid_string_t uuidStr;
  uuid_unparse_lower(uuid, uuidStr);
  NSString *result = [NSString stringWithUTF8String:uuidStr];
  return [[result stringByReplacingOccurrencesOfString:@"-" withString:@""] lowercaseString];
}

- (NSNumber *)systemTimestampMs {
  return @((long long)([[NSDate date] timeIntervalSince1970] * 1000));
}

- (void)createPasskey:(NSString *)rpId userName:(NSString *)userName resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject {
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

- (NSArray<NSString *> *)getPasskeyList {
  return @[];
}

- (void)authenticatePasskey:(NSString *)rpId challenge:(NSString *)challenge resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject {
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

- (NSNumber *)deletePasskey:(NSString *)credentialId {
  return @NO;
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