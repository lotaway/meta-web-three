/**
 * Feature Flags — 功能开关中心
 *
 * 所有开关均从 Expo 环境变量（EXPO_PUBLIC_*）读取，
 * 在 .env.example 中均有对应说明与默认值。
 *
 * 注意：修改 .env 后需重启 Metro（npx expo start --clear）才能生效。
 */

/**
 * Passkey (WebAuthn) 无密码认证功能开关。
 *
 * 默认：false（关闭）
 * 启用条件：
 *   1. 将 EXPO_PUBLIC_PASSKEY_ENABLED=true 写入本地 .env
 *   2. 设备运行 iOS 16+ / Android 9+
 *   3. 后端 passkey-service 已部署并可达
 *   4. EXPO_PUBLIC_RP_ID 已配置为正确的 Relying Party 域名
 *
 * 关闭时效果：
 *   - profile 页不渲染 PasskeyAuthDemo 区块及菜单入口
 *   - /passkey-demo 路由直接返回占位页，不加载任何 Passkey 逻辑
 *   - usePasskey hook 所有方法均为无操作，不发起任何网络或设备请求
 */
export const FEATURE_PASSKEY_ENABLED: boolean =
  process.env.EXPO_PUBLIC_PASSKEY_ENABLED === 'true'
