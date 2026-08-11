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

### 1.1 公共模块 — TenantContext + TenantAwareDO ✅ 已通过审查
- TenantContext (ThreadLocal holder + clear())
- TenantAwareDO (extends BaseDO, 含 tenantId 字段)

### 1.2 公共模块 — 请求头 + JWT + 网关传播 ✅ 已通过审查
- HeaderConstants.X-Tenant-Id / RequestHeaderKeys.TENANT_ID
- UserTokenClaims.tenantId + UserJwtUtil claim 编解码
- UserAuthFilter 传播 X-Tenant-Id 到下游

### 1.3 公共模块 — Filter + ErrorCode + MyBatis 配置 ✅ 已通过审查
- TenantContextFilter (try/finally 清理) + ResponseStatus 1101-1103/1206
- MultiTenantMybatisConfig (TenantLineInnerInterceptor, ignoreTable 按 getTenantTables)

### 1.4 新建 tenant-service 微服务 ✅ 已通过审查并修复
结构齐全（pom/application/schema/实体/Mapper/Service/Controller/proto）。此前发现的问题已修复：
- [x] tenant-service: 网关路由 `/tenant-service/**` 曾仅允许 MERCHANT（ADMIN 无法访问）→ GatewayAuthConfig 已拆分为 `/tenant-service/tenant/*/approve|reject|disable`→ADMIN、`/tenant-service/tenant/**`→ADMIN+MERCHANT，`matchRoute` 升级为 AntPathMatcher 最长匹配
- [x] tenant-service: 注册/验证码/邮箱接口不可达 → `/tenant-service/tenant/captcha/**`、`/tenant-service/tenant/email/**`、`/tenant-service/tenant/register` 已加入 gateway excluded-path-patterns 与 rbac-excluded-path-patterns
- [x] tenant-service: 管理接口无权限校验 → TenantController 的 create/update/getById/getByCode/page/approve/reject/disable/removeUser 已加 ADMIN-only 校验，shop/users 接口已加 MERCHANT/ADMIN 校验（X-User-Role header）

### 1.5 服务注册与部署 ✅ 已通过审查
- server/pom.xml 模块、scripts/server-services-registry.sh、server/Dockerfile stage、docker-compose.server.yml、allow-ports-firework.sh(10126)

### 1.6 BaseEvent 增加 tenantId ✅ 已通过审查
- event-sdk: BaseEvent 含 tenantId 字段

## Phase 2: 商城域启用租户隔离 (mall-domain)

> 审查结论：7 个服务均正确继承 MultiTenantMybatisConfig 且 V2 迁移与 getTenantTables 一致。此前标记的"实体改继承 TenantAwareDO"经核实为误判：TenantLineInnerInterceptor 在 SQL 层注入 tenant_id，不依赖实体继承；且直接继承 BaseDO 会使 createdAt/updatedAt fill 写入各表不存在的列导致破坏。实体保留手动 tenantId 字段（功能正确），此条关闭。

### 2.1 product-service ✅ 已通过审查
- [x] product-service: 表加 tenant_id 列 (V2__add_tenant_id.sql) ✅
- [x] product-service: 实体 tenantId 字段（手动声明，拦截器 SQL 层隔离）✅
- [x] product-service: 启用 MultiTenantMybatisConfig ✅
- 说明: tb_goods_gallery / tb_product_limits / tb_product_stats / tb_product_entity 未在本仓库 schema 定义（外部管理表），无法安全加 tenant_id 列，保持豁免并已记录

### 2.2 order-service ✅ 已通过审查并修复
- [x] order-service: 表加 tenant_id 列 ✅
- [x] order-service: 实体 tenantId 字段 ✅
- [x] order-service: 启用 MultiTenantMybatisConfig ✅
- [x] order-service: tb_company_address（商家公司地址）已加入隔离（V2 迁移加列 + 实体字段 + getTenantTables），该表由 HTTP Controller 在租户上下文内访问 ✅
- 说明: tb_order_setting 为全局订单配置（"global order setting"），豁免合理；tb_order_operate_log 无代码引用（schema 残留表），不处理

### 2.3 cart-service ✅ 已通过审查
- [x] cart-service: 表加 tenant_id 列 ✅
- [x] cart-service: 实体 tenantId 字段 ✅
- [x] cart-service: 启用 MultiTenantMybatisConfig ✅

### 2.4 promotion-service ✅ 已通过审查
- [x] promotion-service: 表加 tenant_id 列 ✅
- [x] promotion-service: 实体 tenantId 字段 ✅
- [x] promotion-service: 启用 MultiTenantMybatisConfig ✅
- 说明: tb_coupon_product_relation 无代码引用（仅 schema.sql 定义），不处理；flash/cms/home 系列为运营配置表，保持全局

### 2.5 payment-service ✅ 已通过审查
- [x] payment-service: 表加 tenant_id 列（Exchange_Orders/User_Kyc/Credit_Profile/payment_reconciliation_diff 隔离；Crypto_Prices 全局豁免）✅
- [x] payment-service: 实体继承 TenantAwareDO（UserKYC/ExchangeOrder/ReconciliationDiffDO 已继承）✅
- [x] payment-service: 启用 MultiTenantMybatisConfig ✅

### 2.6 after-sale-service ✅ 已通过审查
- [x] after-sale-service: 表加 tenant_id 列 ✅
- [x] after-sale-service: 实体 tenantId 字段 ✅
- [x] after-sale-service: 启用 MultiTenantMybatisConfig ✅

### 2.7 review-service ✅ 已通过审查
- [x] review-service: 表加 tenant_id 列 ✅
- [x] review-service: 实体 tenantId 字段 ✅
- [x] review-service: 启用 MultiTenantMybatisConfig ✅

## Phase 3: 商家入驻功能

### 3.1 商家入驻 ✅ 已通过审查并修复
- [x] tenant-service: 商家注册 API（含 CAPTCHA + 邮箱验证）✅
- [x] tenant-service: IP 限流 — 已实现 IpRateLimitService（Redis INCR 按客户端 IP 计数，窗口与阈值取自 service-governance.rate-limiter.tenantRegister 配置），应用于 register/captcha/email 接口 ✅
- [x] tenant-service: 商家审批流程 (PENDING → APPROVED/REJECTED) ✅
- [x] tenant-service: 商家店铺管理 API ✅
- [x] tenant-service: 修复实体类缺少 @TableId id 字段问题 ✅

### 3.2 网关角色 ✅ 已通过审查
- gateway: GatewayAuthConfig 增加 MERCHANT 角色 + 路由规则（含 1.4 已修复的 `/tenant-service/**` 角色映射）

## Phase 4: 前端管理后台 (backstage-admin) ✅ 已通过审查
- admin: 租户 API 模块 (src/apis/tenant.ts)
- admin: 租户列表 + 详情/审批 + 店铺管理页面 (src/views/tenant/index.vue)
- admin: SERVICE_PREFIX_MAP 增加 tenant-service
- admin: 路由注册 asyncRouterMap

## Phase 5: 验证 ✅ 已通过审查
- verify: git diff 确认 ERP/SC/MES 代码未被修改
- verify: 编译通过 (mvn compile)
- verify: TenantContextFilter ThreadLocal 正确清理

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
