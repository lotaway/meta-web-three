# 检查项

## 前端 @directory:client/ 是否按照 @directory:temp/mall-app-web/ 完成所有功能迁移
- [ ] **待检查**: 需要对比 client/ 与 temp/mall-app-web/ 的页面和功能

## 后端 @directory:server/ 是否按照 @directory:temp/mall/ 完成所有功能迁移
- [ ] **待检查**: 需要对比 server/ 与 temp/mall/ 的后端接口和业务逻辑

## 后台管理 @directory:backstage-admin/ 是否按照 @directory:temp/mall-admin-web/ 完成所有功能迁移

## 检查后台 @directory:temp/mall-admin-web 是否对接了后端 @directory:server/ 所需的管理功能

### 秒杀与限时购功能模块
- [ ] **未完成**: 消费者端闪购购买流程（首页展示、闪购价下单、库存扣减、并发保护）
- [ ] **未完成**: 无面向消费者的闪购 API（获取当前活动、场次商品、闪购下单）

### 后台管理与管理员权限
- [x] **已修复**: 密码 BCrypt 加密存储，新增修改密码端点
- [ ] **待修复**: Token 刷新和注销（logout 为空实现）

### MinIO/Media 图片上传功能与权限
- [ ] **待实现**: 基于用户角色的上传限制（管理员无限制，普通用户限制单次大小和总配额）
- [ ] **待实现**: 单次上传大小按角色区分（普通用户 10MB，商家 50MB 等）
- [ ] **待实现**: 用户存储配额跟踪（需要新数据库表）
