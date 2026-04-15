#import "Appsdk.h"
#import <React/RCTBridgeModule.h>
#import <React/RCTBridge.h>
#import <CommonCrypto/CommonDigest.h>
#import <CommonCrypto/CommonHMAC.h>

@implementation Appsdk

RCT_EXPORT_MODULE()

- (dispatch_queue_t)methodQueue {
  return dispatch_get_main_queue();
}



- (NSString *)generateRequestSignature:(NSDictionary *)params secretKey:(NSString *)secretKey {
  NSArray *sortedKeys = [[params allKeys] sortedArrayUsingSelector:@selector(compare:)];
  NSMutableArray *pairs = [NSMutableArray array];
  for (NSString *key in sortedKeys) {
    [pairs addObject:[NSString stringWithFormat:@"%@=%@", key, params[key]]];
  }
  NSString *message = [[pairs componentsJoinedByString:@"&"] stringByAppendingString:secretKey];
  
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  NSData *data = [message dataUsingEncoding:NSUTF8StringEncoding];
  CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
  
  NSMutableString *signature = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
    [signature appendFormat:@"%02x", digest[i]];
  }
  
  return signature;
}

- (NSString *)preciseAmountSum:(NSString *)amountA amountB:(NSString *)amountB {
  NSDecimalNumber *a = [NSDecimalNumber decimalNumberWithString:amountA];
  NSDecimalNumber *b = [NSDecimalNumber decimalNumberWithString:amountB];
  NSDecimalNumber *sum = [a decimalNumberByAdding:b];
  return sum.stringValue;
}

- (NSString *)computeOrderTotal:(NSString *)unitPrice quantity:(double)quantity discountAmount:(NSString *)discountAmount shippingFee:(NSString *)shippingFee {
  NSDecimalNumber *price = [NSDecimalNumber decimalNumberWithString:unitPrice];
  NSDecimalNumber *subtotal = [price decimalNumberByMultiplyingBy: [NSDecimalNumber decimalNumberWithMantissa:quantity exponent:0 isNegative:YES]];
  NSDecimalNumber *discount = [NSDecimalNumber decimalNumberWithString:discountAmount];
  NSDecimalNumber *shipping = [NSDecimalNumber decimalNumberWithString:shippingFee];
  NSDecimalNumber *total = [[subtotal decimalNumberBySubtracting:discount] decimalNumberByAdding:shipping];
  return total.stringValue;
}

- (NSString *)hmacSign:(NSString *)message signingKey:(NSString *)signingKey {
  const char *cKey  = [signingKey cStringUsingEncoding:NSASCIIStringEncoding];
  const char *cData = [message cStringUsingEncoding:NSASCIIStringEncoding];
  unsigned char cHMAC[CC_SHA256_DIGEST_LENGTH];
  CCHmac(kCCHmacAlgSHA256, cKey, strlen(cKey), cData, strlen(cData), cHMAC);
  
  NSData *hash = [[NSData alloc] initWithBytes:cHMAC length:CC_SHA256_DIGEST_LENGTH];
  NSMutableString *result = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
    [result appendFormat:@"%02x", ((unsigned char*)hash.bytes)[i]];
  }
  return result;
}

- (NSString *)createNonce {
  CFUUIDRef uuid = CFUUIDCreate(kCFAllocatorDefault);
  CFStringRef uuidString = CFUUIDCreateString(kCFAllocatorDefault, uuid);
  NSString *nonce = (__bridge NSString *)uuidString;
  CFRelease(uuidString);
  CFRelease(uuid);
  return [nonce stringByReplacingOccurrencesOfString:@"-" withString:@""];
}

- (NSNumber *)systemTimestampMs {
  return @([[NSDate date] timeIntervalSince1970] * 1000);
}

- (void)createPasskey:(NSString *)rpId userName:(NSString *)userName resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject {
  NSString *passkey = [NSString stringWithFormat:@"passkey-%@", [self createNonce]];
  resolve(passkey);
}

- (NSArray *)getPasskeyList {
  return @[@"passkey-1", @"passkey-2"];
}

- (void)authenticatePasskey:(NSString *)challenge resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject {
  resolve(@YES);
}

- (void)deletePasskey:(NSString *)credentialId {
}

@end

