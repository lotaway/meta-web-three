# RN 客户端 TODO

## 个人中心未完成功能

### 1. 个人中心页面 (profile.tsx)
- [ ] 用户信息未从 API 获取，当前为硬编码 Mock 数据
  - 昵称 (nickname): 固定为 "游客"
  - 积分 (integration): 0
  - 成长值 (growth): 0
  - 优惠券数量 (couponCount): 0
- [ ] 需要接入用户详情 API (如 `/user/detail` 或类似)

### 2. 分类页面 (category.tsx)
- [ ] 分类图标使用 placeholder.com 占位图
- [ ] 需确认 CategoryContainer 是否正确连接 API

### 3. 首页 (index.tsx)
- [ ] 需检查商品列表是否从 API 获取

### 4. 其他
- [ ] Passkey 登录成功后 token 未存储到全局状态
- [ ] 缺少用户认证状态管理

## 模块架构迁移
- [ ] 将 `client/turbo-module/wechat-pay` Turbo Module 迁移为 Expo Module
  - 当前为 Turbo Module（`create-react-native-library` + `codegenConfig` + `react-native-builder-bob`）
  - 目标：参考 `client/modules/appsdk` 或 `client/modules/scanner-module` 改造为 Expo Module
  - 需添加 `expo-module.config.json`，使用 `expo-module-scripts` 构建
  - 需替换 `NativeEventEmitter`/`NativeModules` 为 Expo 的 `requireNativeModule` API
  - 需调整 `package.json` 的 scripts、dependencies 和 peerDependencies（引入 `expo`）
  - 需迁移 Android/iOS 原生代码到 Expo Module 目录结构
