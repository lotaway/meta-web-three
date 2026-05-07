# 电商App功能补全 TODO

## 前端所需功能补全（对比 directory:/temp/mall-app-web/ 缺失项）

### 搜索模块接口补全（整个模块缺失）
- [x] 17.4 `POST /esProduct/importAll` - 导入商品到 ES
- [x] 17.5 `POST /esProduct/create/{id}` - 同步商品到 ES

---

## 后端接口补全（对比 directory:/temp/mall/ 缺失项）

### 搜索接口

- [x] `POST /esProduct/importAll` - 导入商品到 ES
- [x] `POST /esProduct/create/{id}` - 同步商品到 ES

---

## P0 - 核心功能（必须完成）

### 1. 用户注册/登录页面
- [ ] 1.6 第三方登录（微信/Apple）

---

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
