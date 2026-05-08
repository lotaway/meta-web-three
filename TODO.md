# 检查项

## 前端 @directory:client/ 是否按照 @directory:temp/mall-app-web/ 完成所有功能迁移

## 后端 @directory:server/ 是否按照 @directory:temp/mall/ 完成所有功能迁移

## 检查后台 @directory:temp/mall-admin-web 是否对接了后端 @directory:server/ 所需的管理功能
- [ ] 验证现有 Admin/Role/Product/Order 管理接口在微服务架构下的兼容性，普通用户鉴权是在 @directory:server/gateway 网关服务里完成的，那管理员鉴权是否也做好了相应处理 @file:ADMIN_AUTH.md