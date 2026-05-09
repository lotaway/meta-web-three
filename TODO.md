# 检查项

## 前端 @directory:client/ 是否按照 @directory:temp/mall-app-web/ 完成所有功能迁移

## 后端 @directory:server/ 是否按照 @directory:temp/mall/ 完成所有功能迁移

### 未迁移的 API 接口（需补充）

| 参考项目路径 | 文件位置 | 接口名称 | 用途 | 当前状态 |
|------------|---------|---------|------|---------|
| `/aliyun/oss/policy` | temp/mall/mall-admin/src/.../OssController.java | 阿里云OSS签名策略 | 获取OSS上传签名 | ❌ 未实现 |
| `/aliyun/oss/callback` | temp/mall/mall-admin/src/.../OssController.java | 阿里云OSS回调 | 处理OSS上传回调 | ❌ 未实现 |
| `/minio/upload` | temp/mall/mall-admin/src/.../MinioController.java | MinIO文件上传 | 通过MinIO上传文件 | ⚠️ 用 /media 替代 |
| `/minio/delete` | temp/mall/mall-admin/src/.../MinioController.java | MinIO文件删除 | 通过MinIO删除文件 | ⚠️ 用 /media 替代 |

**说明**：
- 参考项目使用 MinIO 作为对象存储，当前项目使用本地文件系统存储（/media 路径）
- 阿里云 OSS 相关接口未实现，如需支持阿里云 OSS 需要补充

## 检查后台 @directory:temp/mall-admin-web 是否对接了后端 @directory:server/ 所需的管理功能