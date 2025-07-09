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
