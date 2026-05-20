# Wechat Pay Module

Expo Module for Wechat Pay integration.

## Installation

```bash
expo install wechat-pay-module
```

## Usage

```typescript
import WechatPayModule from 'wechat-pay-module';

// Initialize
WechatPayModule.init('your-wechat-app-id');

// Check if Wechat is installed
const installed = await WechatPayModule.isWechatInstalled();

// Make payment
await WechatPayModule.pay({
  partnerId: '...',
  prepayId: '...',
  nonceStr: '...',
  timeStamp: '...',
  packageValue: '...',
  sign: '...',
});
```

## Platform Support

| Platform | Support |
|----------|---------|
| iOS      | ✅     |
| Android  | ✅     |
| Web      | ❌     |

## License

MIT
