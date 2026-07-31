# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

- [ ] 当前 `/developer/register` 接口存在安全隐患：
  - 任何人都可以无限制调用
  - 没有 CAPTCHA、没有邮箱验证、没有 IP 限流
  - 唯一的防护是注册后需要管理员人工审批（PENDING → APPROVED）
  对于 B2B 场景，更安全的做法是：
  1. __不暴露公开注册 API__，改为通过商务渠道线下收集开发者信息，由管理员在 backstage-admin 中创建账号
  2. 如果必须保留公开注册，至少加上 __CAPTCHA + 邮箱验证 + IP 限流__ 三层防护


# 多租户 SaaS 架构 (Multi-Tenant)

## Phase 1: 基础设施 (Backend Infrastructure)

### 1.1 公共模块 — TenantContext + TenantAwareDO
- [x] common: 创建 TenantContext (ThreadLocal holder)
- [x] common: 创建 TenantAwareDO (extends BaseDO, 增加 tenantId 字段)

### 1.2 公共模块 — 请求头 + JWT + 网关传播
- [x] common: HeaderConstants 增加 X-Tenant-Id
- [x] common: RequestHeaderKeys 枚举增加 TENANT_ID
- [x] gateway: UserTokenClaims record 增加 tenantId 字段
- [x] common: UserJwtUtil 增加 tenantId claim 编解码
- [x] gateway: UserAuthFilter 传播 X-Tenant-Id 到下游服务

### 1.3 公共模块 — Filter + ErrorCode + MyBatis 配置
- [x] common: 创建 TenantContextFilter (Servlet Filter, 提取 X-Tenant-Id 到 TenantContext)
- [x] common: ResponseStatus 增加租户错误码 (1101-1103)
- [x] common: 创建 MultiTenantMybatisConfig (TenantLineInnerInterceptor)

### 1.4 新建 tenant-service 微服务
- [x] tenant-service: pom.xml + application.yml + BaseApplication
- [x] tenant-service: db/schema.sql (tenant/tenant_shop/tenant_user 表)
- [x] tenant-service: 实体类 (Tenant/TenantShop/TenantUser)
- [x] tenant-service: Mapper 接口 (TenantMapper/TenantShopMapper/TenantUserMapper)
- [x] tenant-service: Service 接口 + 实现 (TenantService)
- [x] tenant-service: Controller (CRUD + 注册 + 审批 + 店铺管理 + 用户关联)
- [x] protos: TenantService.proto 定义

### 1.5 服务注册与部署
- [x] server/pom.xml: 添加 tenant-service 模块
- [x] scripts: server-services-registry.sh 注册 tenant-service
- [x] server/Dockerfile: 增加 tenant-service build stage
- [x] docker-compose.server.yml: 增加 tenant-service service
- [x] allow-ports-firework.sh: 追加端口 10126

### 1.6 BaseEvent 增加 tenantId
- [x] event-sdk: BaseEvent 增加 tenantId 字段

## Phase 2: 商城域启用租户隔离 (mall-domain)

### 2.1 product-service
- [x] product-service: 表加 tenant_id 列 (V2__add_tenant_id.sql)
- [x] product-service: 实体改继承 TenantAwareDO
- [x] product-service: 启用 MultiTenantMybatisConfig

### 2.2 order-service
- [x] order-service: 表加 tenant_id 列
- [x] order-service: 实体改继承 TenantAwareDO
- [x] order-service: 启用 MultiTenantMybatisConfig

### 2.3 cart-service
- [x] cart-service: 表加 tenant_id 列
- [x] cart-service: 实体改继承 TenantAwareDO
- [x] cart-service: 启用 MultiTenantMybatisConfig

### 2.4 promotion-service
- [x] promotion-service: 表加 tenant_id 列
- [x] promotion-service: 实体改继承 TenantAwareDO
- [x] promotion-service: 启用 MultiTenantMybatisConfig

### 2.5 payment-service
- [ ] payment-service: 表加 tenant_id 列（暂缓 — 需要评估支付流水对租户隔离的影响）
- [ ] payment-service: 实体改继承 TenantAwareDO
- [ ] payment-service: 启用 MultiTenantMybatisConfig

### 2.6 after-sale-service
- [x] after-sale-service: 表加 tenant_id 列
- [x] after-sale-service: 实体改继承 TenantAwareDO
- [x] after-sale-service: 启用 MultiTenantMybatisConfig

### 2.7 review-service
- [x] review-service: 表加 tenant_id 列
- [x] review-service: 实体改继承 TenantAwareDO
- [x] review-service: 启用 MultiTenantMybatisConfig

## Phase 3: 商家入驻功能

### 3.1 商家入驻
- [x] tenant-service: 商家注册 API (含 CAPTCHA + 邮箱验证 + IP 限流)
- [x] tenant-service: 商家审批流程 (PENDING → APPROVED/REJECTED)
- [x] tenant-service: 商家店铺管理 API
- [x] tenant-service: 修复实体类缺少 @TableId id 字段问题

### 3.2 网关角色
- [x] gateway: GatewayAuthConfig 增加 MERCHANT 角色 + 路由规则

## Phase 4: 前端管理后台 (backstage-admin)

### 4.1 租户管理页面
- [x] admin: 租户 API 模块 (src/apis/tenant.ts)
- [x] admin: 租户列表 + 详情/审批 + 店铺管理页面 (src/views/tenant/index.vue)
- [x] admin: SERVICE_PREFIX_MAP 增加 tenant-service
- [x] admin: 路由注册 asyncRouterMap

## Phase 5: 验证
- [x] verify: git diff 确认 ERP/SC/MES 代码未被修改 ✅
- [ ] verify: 编译通过 (mvn compile)
- [ ] verify: TenantContextFilter ThreadLocal 正确清理

## [SOLANA商品合约](./TODO_SOLANA.md)

## [区块重组处理](./TODO_BLOCK_REORG.md)

# 待决议功能

- [ ] 实现事件溯源和CQRS模式 (Event Sourcing + CQRS，提升数据一致性和审计能力)

- [ ] 添加无服务器函数计算 (Serverless，应对突发高并发场景，如秒杀活动)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加区块链集成 (供应链透明化、NFT会员积分、商品溯源)

- [ ] 实现AR/VR购物体验 (3D商品展示、虚拟试穿、沉浸式购物)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)