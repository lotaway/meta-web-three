// Reexport the native module. On web, it will be resolved to WechatPayModule.web.ts
// and on native platforms to WechatPayModule.ts
export { default } from './WechatPayModule';
export * from './WechatPayModule.types';
