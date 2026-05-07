# 检查项

## 前端 @directory:/client/ 是否按照 @directory:/temp/mall-app-web/ 完成所有功能迁移

## 后端 @directory:/server/ 是否按照 @directory:/temp/mall/ 完成所有功能迁移

## 检查后台 @directory:/temp/mall-admin-web 是否对接了后端 @directory:/server/ 所需的管理功能

## 技术债务

### 后端技术债务
- [ ] 统一错误处理（全局异常处理器）
- [ ] 服务间调用优化（RestTemplate → WebClient）
- [x] 商品 ES 搜索实现
- [ ] 订单自动取消定时任务（Quartz/Spring Scheduled）
- [ ] 支付宝异步回调完整实现（生产环境配置）

### 前端技术债务
- [ ] 统一错误处理（API 调用错误拦截）
- [ ] 网络请求拦截器优化（Axios 拦截器）
- [ ] 图片缓存优化（react-native-fast-image）
- [ ] 性能优化（FlatList 优化/内存优化）
- [ ] 离线缓存策略（React Query/Apollo Cache）
- [ ] 单元测试补充（Jest/React Native Testing Library）
- [ ] E2E 测试补充（Detox/Appium）

---
