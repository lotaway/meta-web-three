# JWT 认证实现说明

## 概述

本项目在用户服务（user-service）中实现了JWT（JSON Web Token）的生成和验证功能。JWT在用户登录成功后生成，用于后续的API认证。

## 架构设计

### JWT生成位置
- **用户服务（user-service）**: 负责JWT的生成
- **网关服务（gateway）**: 负责JWT的验证和解析

### 核心组件

1. **JwtUtil** (`server/common/src/main/java/com/metawebthree/common/utils/JwtUtil.java`)
   - JWT工具基类
   - 提供JWT生成、解析、验证功能
   - 支持自定义claims和过期时间

2. **UserJwtUtil** (`server/gateway/src/main/java/com/metawebthree/utils/UserJwtUtil.java`)
   - 继承自JwtUtil
   - 专门处理用户相关的JWT操作
   - 包含用户角色和权限信息

3. **UserAuthFilter** (`server/gateway/src/main/java/com/metawebthree/Filters/UserAuthFilter.java`)
   - 网关过滤器
   - 拦截请求并验证JWT
   - 解析用户信息并添加到请求头

## 登录接口

### 1. 邮箱密码登录

**接口**: `POST /user/signIn`

**参数**:
- `email`: 用户邮箱
- `password`: 用户密码
- `typeId`: 用户类型（可选，默认0）

**返回**:
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": 1,
      "email": "user@example.com",
      "typeId": 0
    },
    "walletAddress": null,
    "loginType": "email"
  }
}
```

### 2. Web3钱包登录

**接口**: `POST /user/signInWithWallet`

**参数**:
- `walletAddress`: 钱包地址
- `timestamp`: 时间戳
- `signature`: 签名

**返回**:
```json
{
  "status": 200,
  "message": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": 2,
      "email": "0x1234...@wallet.local",
      "typeId": 1,
      "walletAddress": "0x1234..."
    },
    "walletAddress": "0x1234...",
    "loginType": "wallet"
  }
}
```

## JWT Token结构

### Header
```json
{
  "JWT": "JWT",
  "alg": "HS256",
  "typ": "JWT"
}
```

### Payload
```json
{
  "sub": "用户ID",
  "userId": "用户ID",
  "email": "用户邮箱",
  "typeId": "用户类型",
  "role": "用户角色",
  "walletAddress": "钱包地址（仅钱包登录）",
  "iat": "签发时间",
  "exp": "过期时间"
}
```

## 配置

### JWT密钥配置
在 `application-common.yml` 中配置JWT密钥：
```yaml
jwt:
  secret: mwt123123
```

### 数据库表结构
User表需要包含以下字段：
```sql
CREATE TABLE User (
  id BIGSERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  password VARCHAR(255),
  author_id INT,
  type_id SMALLINT DEFAULT 0,
  wallet_address VARCHAR(255)
);
```

## 使用流程

1. **用户登录**: 调用登录接口，验证用户凭据
2. **生成JWT**: 登录成功后，用户服务生成JWT token
3. **返回Token**: 将JWT token返回给客户端
4. **客户端存储**: 客户端保存JWT token（通常在localStorage或sessionStorage中）
5. **API调用**: 客户端在后续API请求中携带JWT token
6. **网关验证**: 网关过滤器验证JWT token的有效性
7. **请求转发**: 验证通过后，网关将请求转发到相应的微服务

## 安全考虑

1. **密钥管理**: JWT密钥应该安全存储，不应硬编码在代码中
2. **Token过期**: 设置合理的token过期时间（当前为30天）
3. **HTTPS**: 生产环境必须使用HTTPS传输JWT
4. **刷新Token**: 考虑实现refresh token机制
5. **Token撤销**: 实现token撤销机制（如用户登出时）

## 扩展功能

1. **角色权限**: 可以在JWT中包含用户角色和权限信息
2. **多租户**: 支持多租户场景下的JWT
3. **设备管理**: 支持多设备登录和设备管理
4. **审计日志**: 记录JWT的生成和使用日志

## 故障排除

### 常见问题

1. **Token过期**: 检查token的过期时间设置
2. **密钥不匹配**: 确保用户服务和网关使用相同的JWT密钥
3. **签名验证失败**: 检查JWT的签名算法和密钥
4. **用户信息解析错误**: 检查JWT payload中的用户信息格式

### 调试方法

1. 使用JWT调试工具（如jwt.io）解析token
2. 检查网关日志中的JWT验证信息
3. 验证数据库中的用户信息是否正确
4. 确认配置文件中的JWT设置 
