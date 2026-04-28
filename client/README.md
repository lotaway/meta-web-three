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

## 获取iOS设备UDID用于预览应用

[https://www.betaqr.com.cn/udid](https://www.betaqr.com.cn/udid)

## 发布

### iOS 证书和描述文件

Expo EAS 通过 Apple Developer API 自动管理证书和描述文件，无需手动下载/配置。实现方式如下：

当你在终端运行 eas build:configure 或首次执行 eas build --platform ios 时，Expo CLI 会要求你：

- 登录 Apple Developer 账号（需 Account Holder 或 Admin 权限）。
- 开启 App Store Connect API 访问权限（需在 Apple Developer 中生成 API Key）。
- 授权 Expo 使用你的团队 ID（appleTeamId）。

EAS 服务器会通过 Apple API 自动完成以下操作：

- 创建所需的 Development/Distribution Certificates
- 生成匹配的 Provisioning Profiles
- 处理推送证书（需部分手动操作，见下文）
- 需设置环境变量 EXPO_APPLE_PASSWORD 和 EXPO_APPLE_APP_SPECIFIC_PASSWORD 实现 CI 自动化

### 如何生成apk

安卓默认打包使用`yarn deploy:android`命令生成谷歌的aab文件，如果需要生成apk文件有两种方式：
方式一，修改`eas.json`里的build方式然后重新打包：

```json
"build": {
    "android": {
      "buildType": "apk",
      "gradleCommand": ":app:assembleRelease"
    }
  }
```

方式二，使用转换工具，安装并配置好[bundletool](https://github.com/google/bundletool)，然后在终端运行：

```bash
bundletool build-apks --bundle=build.aab --output=app.apks --mode=universal --ks=./keys/android.jks --ks-key-alias=android-keys --ks-pass=pass:your-keystore-password --key-pass=pass:your-key-password
```

打包出来的是谷歌的apks内含多文件，使用解压命令得到apk：

```bash
unzip app.apks
```

也可以使用命令直接安装apks到本地设备：

```bash
bundletool install-apks --apks=app.apks
```

## 代码生成

### Generate API Interfaces

This method relies on the OpenAPI interface documentation and generator. The specific script is located at `tools/OpenapiToTS.js`. Configure `NEXT_PUBLIC_BACK_API_DOC_HOST` or `NEXT_PUBLIC_BACK_API_HOST` in the `.env` file to point to the OpenAPI doc configuration URL, then run the following command to generate the encapsulated API calls:

```bash
yarn generate:api
```

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

## 支付模块配置

### 支付方式

项目包含三种支付方式：

| 模块 | 说明 | 目录 |
|------|------|------|
| 微信支付 | TurboModule | `turbo-module/wechat-pay/` |
| 支付宝 | TurboModule | `turbo-module/alipay/` |
| Stripe | 官方 RN SDK | `@stripe/stripe-react-native` |
| 统一入口 | 统一调用接口 | `app/lib/payment/` |

### 配置步骤

#### 1. 安装依赖

```bash
npm install @stripe/stripe-react-native
```

#### 2. app.json 插件配置

在 `app.json` 的 `expo.plugins` 中添加支付插件：

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

#### 3. iOS SDK 配置（必需）

iOS 官方 SDK（微信、支付宝）**已改为自动下载管理**，通过 `scripts/download-ios-sdks.js` 锁定版本号并统一分发，避免手动下载导致的版本不一致问题。

**自动下载 SDK**
```bash
# 运行脚本自动下载、校验、放置 SDK
yarn setup:ios-sdks
```

脚本行为：
1. 读取 `scripts/download-ios-sdks.js` 中锁定的版本号和下载地址
2. 下载 SDK 压缩包并缓存到 `.ios-sdk-cache/`（避免重复下载）
3. 校验 SHA256（如已配置）
4. 解压并放置 `.xcframework` 到对应 `turbo-module/*/ios/` 目录

> ⚠️ **注意**：微信/支付宝官方不提供直链下载，首次使用前请联系运维/管理员配置内部 CDN 地址。可通过环境变量覆盖：
> ```bash
> WECHAT_SDK_URL=https://your-cdn.com/WeChatOpenSDK.zip \
> ALIPAY_SDK_URL=https://your-cdn.com/AlipaySDK.zip \
>   yarn setup:ios-sdks
> ```

**重新安装依赖**
```bash
cd ios && pod install && cd ..
```

**Stripe SDK** 已通过 npm 安装，无需额外配置。

**为什么不像 Android 一样用包管理器？**

| 平台 | 管理方式 | 原因 |
|------|----------|------|
| Android | Gradle 自动依赖 | Maven Central 上有官方发布的 AAR |
| iOS | 脚本自动下载 | 微信/支付宝未发布到 CocoaPods/SPM，仅提供官网下载 |

**为什么 SDK 不提交到 Git？**
1. **版权限制**：闭源 SDK 的许可证通常禁止二次分发
2. **仓库体积**：`.xcframework` 体积较大，会急剧膨胀 Git 仓库
3. **版本锁定**：通过脚本中的 `version` + `sha256` 字段，比提交二进制文件更能精确控制版本

#### 4. Android SDK 配置

Android SDK 通过 Gradle 自动下载依赖，无需手动配置：

| 支付方式 | Gradle 依赖 |
|---------|------------|
| 微信支付 | `com.tencent.mm.opensdk:wechat-sdk-android:6.8.0` |
| 支付宝 | `com.alipay.sdk:alipaysdk:15.8.0` |

如需更新版本，修改对应模块的 `build.gradle` 文件。

### 支付页面

支付入口页面：`app/checkout.tsx`

流程：
```
购物车 → 结算页(/checkout) → 选择支付方式 → 获取支付参数 → 调起支付 → 验证结果
```

### 注意事项

1. **接口必需**：后端必须实现上述接口，否则无法完成支付
2. **支付结果必须验证**：客户端收到支付成功回调后，必须调用 `/api/pay/verify` 验证
3. **iOS URL Scheme**：微信 `wx` + AppID，支付宝 `alipay` + AppID
4. **Android 回调 Activity**：微信支付需要 `WXPayEntryActivity`，支付宝自动处理
5. **Universal Link**：微信支付需要配置应用关联的 Universal Link
