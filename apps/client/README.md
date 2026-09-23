# Welcome to your Expo app 👋

This is an [Expo](https://expo.dev) project created with [`create-expo-app`](https://www.npmjs.com/package/create-expo-app).

## Get started

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
   npx expo start
   ```

In the output, you'll find options to open the app in a

- [development build](https://docs.expo.dev/develop/development-builds/introduction/)
- [Android emulator](https://docs.expo.dev/workflow/android-studio-emulator/)
- [iOS simulator](https://docs.expo.dev/workflow/ios-simulator/)
- [Expo Go](https://expo.dev/go), a limited sandbox for trying out app development with Expo

You can start developing by editing the files inside the **app** directory. This project uses [file-based routing](https://docs.expo.dev/router/introduction).

### Local development environment variables

After copying `.env.example` to `.env`, only **one variable matters** for local development:

- `EXPO_PUBLIC_BACK_API_HOST` — backend gateway address on the RN side. Metro only inlines `EXPO_PUBLIC_*` (`NEXT_PUBLIC_*` only applies to the web).
  - When left empty, it is derived automatically from the Expo dev server `hostUri`: iOS simulator falls back to `localhost`, Android emulator falls back to `10.0.2.2`.
  - Android emulator: explicitly set `EXPO_PUBLIC_BACK_API_HOST=http://10.0.2.2:10081`
  - Physical device: use the dev machine's LAN IP, e.g. `EXPO_PUBLIC_BACK_API_HOST=http://192.168.1.110:10081`

All other variables can be left unset for local development:

- `NEXT_PUBLIC_BACK_API_DOC_HOST` / `NEXT_PUBLIC_BACK_API_HOST`: used only by the web / code generation workflow.
- `EXPO_PUBLIC_DEFAULT_USER_ID` (default 1), `EXPO_PUBLIC_PASSKEY_ENABLED` (default false), `EXPO_PUBLIC_PASSKEY_RP_ID` (default `localhost`): defaults are already provided in code.
- `EXPO_PUBLIC_WECHAT_APP_ID` / `EXPO_PUBLIC_ALIPAY_APP_ID` / `EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY`: payment-related only, default to empty strings.
- `BACKEND_API_ROOT_DIR` / `CONTRACT_ROOT_DIR`: needed only by the `generate:enum` / `generate:contract:abi` scripts.

## Get a fresh project

When you're ready, run:

```bash
npm run reset-project
```

This command will move the starter code to the **app-example** directory and create a blank **app** directory where you can start developing.

## Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.

## ios module generate

If want to edit ios directory or ios turbo module code, need to use `npm run pod` in project root or use `pod install` inside ios directory to link, generate relative libs and framework, base code.

## Get the iOS device UDID for previewing the app

[https://www.betaqr.com.cn/udid](https://www.betaqr.com.cn/udid)

## Release

### iOS certificates and provisioning profiles

Expo EAS manages certificates and provisioning profiles automatically through the Apple Developer API, so there is no need to download/configure them manually. Here is how it works:

When you run `eas build:configure` or run `eas build --platform ios` for the first time, the Expo CLI will ask you to:

- Sign in with an Apple Developer account (requires Account Holder or Admin permissions).
- Enable App Store Connect API access (an API Key must be generated in Apple Developer).
- Authorize Expo to use your team ID (`appleTeamId`).

The EAS servers then perform the following automatically through the Apple API:

- Create the required Development/Distribution Certificates
- Generate matching Provisioning Profiles
- Handle push certificates (partially manual, see below)
- Set the `EXPO_APPLE_PASSWORD` and `EXPO_APPLE_APP_SPECIFIC_PASSWORD` environment variables for CI automation

### How to generate an APK

Android builds default to the `yarn deploy:android` command, which produces a Google AAB file. If you need an APK, there are two options:
Option 1 — change the build method in `eas.json` and rebuild:

```json
"build": {
    "android": {
      "buildType": "apk",
      "gradleCommand": ":app:assembleRelease"
    }
  }
```

Option 2 — use a conversion tool. Install and configure [bundletool](https://github.com/google/bundletool), then run:

```bash
bundletool build-apks --bundle=build.aab --output=app.apks --mode=universal --ks=./keys/android.jks --ks-key-alias=android-keys --ks-pass=pass:your-keystore-password --key-pass=pass:your-key-password
```

The output is a Google `apks` file containing multiple files; unzip it to get the APK:

```bash
unzip app.apks
```

You can also install the APKS directly to a local device:

```bash
bundletool install-apks --apks=app.apks
```

## Code generation

### Generate API Interfaces

This method relies on the OpenAPI interface documentation and generator. The shared script is located at `apps/tools/OpenApiToTS.js`, used by both `apps/client/` and `apps/backstage-admin/`.

**Workflow:** Download OpenAPI docs → Generate TypeScript-Fetch API client → Inject env variable as base URL.

**Prerequisites:**
- Backend service is running
- Install openapi-generator-cli: `npm install -g @openapitools/openapi-generator-cli`

**Configuration (`.env.development`):**
```bash
NEXT_PUBLIC_BACK_API_HOST=http://localhost:10081
NEXT_PUBLIC_BACK_API_DOC_HOST=http://localhost:10081   # defaults to API_HOST
```

**Run:**
```bash
yarn generate:api
```

**Generated structure:**
```
src/generated/api/
├── apis/           # API classes (one per Controller)
├── models/         # TypeScript interfaces / type definitions
├── runtime.ts      # Base runtime
├── index.ts        # Export entry
└── openapi.json    # Raw OpenAPI spec
```

**Usage:**
```typescript
import { DefaultApi, OrderApi } from '@/generated/api'

const api = new DefaultApi()
const orders = await api.listOrders({ status: 'pending' })

const orderApi = new OrderApi()
await orderApi.createOrder({ order: orderRequest })
```

**Notes:**
- Generated code overwrites all files under `src/generated/api/`
- To customize generation logic, modify `apps/tools/OpenApiToTS.js`
- After first generation, consider committing `openapi.json` for offline development

### Generate Enums

To ensure consistency between frontend and backend enums, generate enum files. The specific script is located at `tools/JEnumToTS.js`. Ensure the backend Java program is placed in the local file directory, then configure `BACKEND_API_ROOT_DIR` in the `.env` file to point to the program directory, and run the following command:

```bash
yarn generate:enum
```

### Generate Contract ABI

To generate and use contract ABI, place the contract ABI files in the contract directory. The specific script is located at `tools/CopyContractABIToTS.js`. Ensure the contract program is placed in the local file directory, then configure `CONTRACT_ROOT_DIR` in the `.env` file to point to the contract directory.

Then, in the contract root directory, run the following command:

```bash
hardhat compile
```

Next, return to the frontend project root directory and run the following command:

```bash
yarn generate:contract:abi
```

## Payment module configuration

### Payment methods

The project supports three payment methods:

| Module | Description | Directory |
|------|------|------|
| WeChat Pay | Expo Module | `modules/wechat-pay-module/` |
| Alipay | Expo Module | `modules/alipay-module/` |
| Stripe | Official RN SDK | `@stripe/stripe-react-native` |
| Unified entry | Unified calling interface | `app/lib/payment/` |

### Configuration steps

#### 1. Install dependencies

```bash
npm install @stripe/stripe-react-native
```

#### 2. Configure the app.json plugin

Add the payment plugin to `expo.plugins` in `app.json`:

```json
{
  "expo": {
    "plugins": [
      ["./plugins/with-payment", {
        "wechatAppId": "wx1234567890",
        "alipayAppId": "2021001234567890"
      }]
    ]
  }
}
```

#### 3. iOS SDK configuration (required)

The official iOS SDKs (WeChat, Alipay) **are now managed through automatic downloads**: `scripts/download-sdks.js` locks the version numbers and distributes them uniformly, avoiding inconsistency caused by manual downloads.

**Download SDKs automatically**
```bash
# Runs the script to automatically download, verify, and place the SDKs
yarn setup:ios-sdks
```

What the script does:
1. Reads the locked version numbers and download URLs from `scripts/download-sdks.js`
2. Downloads the SDK archives and caches them in `.ios-sdk-cache/` (avoids re-downloading)
3. Verifies SHA256 (if configured)
4. Extracts and places the `.xcframework` files into the corresponding `turbo-module/*/ios/` directories

> ⚠️ **Note**: WeChat/Alipay do not provide official direct download links. Before first use, contact operations/admin to configure an internal CDN URL. You can override via environment variables:
> ```bash
> WECHAT_SDK_URL=https://your-cdn.com/WeChatOpenSDK.zip \
> ALIPAY_SDK_URL=https://your-cdn.com/AlipaySDK.zip \
>   yarn setup:ios-sdks
> ```

**Reinstall dependencies**
```bash
cd ios && pod install && cd ..
```

**Stripe SDK** is installed via npm, no extra configuration is needed.

**Why not use a package manager like on Android?**

| Platform | Management | Reason |
|------|----------|------|
| Android | Automatic via Gradle | Official AARs are published on Maven Central |
| iOS | Script-based download | WeChat/Alipay do not publish to CocoaPods/SPM, only provide downloads from their websites |

**Why aren't the SDKs committed to Git?**
1. **Copyright restrictions**: closed-source SDK licenses usually prohibit redistribution
2. **Repository size**: `.xcframework` files are large and would bloat the Git repository
3. **Version locking**: the `version` + `sha256` fields in the script control versions more precisely than committing binaries

#### 4. Android SDK configuration

The Android SDKs are downloaded automatically via Gradle, no manual configuration needed:

| Payment method | Gradle dependency |
|---------|------------|
| WeChat Pay | `com.tencent.mm.opensdk:wechat-sdk-android:6.8.0` |
| Alipay | `com.alipay.sdk:alipaysdk:15.8.0` |

To update versions, modify the corresponding module's `build.gradle` file.

### Payment page

Payment entry page: `app/checkout.tsx`

Flow:
```
Cart → Checkout page (/checkout) → Choose payment method → Get payment params → Invoke payment → Verify result
```

### Notes

1. **Interfaces required**: the backend must implement the interfaces above, otherwise payment cannot complete
2. **Payment results must be verified**: after receiving the payment-success callback, the client must call `/api/pay/verify`
3. **iOS URL Scheme**: WeChat `wx` + AppID, Alipay `alipay` + AppID
4. **Android callback Activity**: WeChat Pay requires `WXPayEntryActivity`; Alipay handles it automatically
5. **Universal Link**: WeChat Pay requires configuring the app's associated Universal Link