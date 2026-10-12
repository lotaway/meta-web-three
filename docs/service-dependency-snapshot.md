# 服务间调用依赖快照（合并前后对照要点）

> 本文件记录合并前后的关键服务间调用路径，用于追溯合并是否完整。数据基于现有代码（gateway application.yml、GraphQL Federation、Dubbo/Feign、前端映射）。

## 1. 网关路由（REST）

网关（gateway，10081）基于 Spring Cloud Gateway + Zookeeper Discovery Locator。关键路由配置见 [`server/gateway/src/main/resources/application.yml`](../server/gateway/src/main/resources/application.yml)。

- Discovery locator：`spring.cloud.gateway.discovery.locator` 将请求按 `lb://<service-id>/<path>` 转发。**服务 id 已统一为合后 15 服务名**（见 [`scripts/server-services-registry.sh`](../scripts/server-services-registry.sh)）。
- 免鉴权/豁免路径、鉴权角色配置：`GatewayAuthConfig` 已按合后服务名与实际 Controller 路径同步（提交 `a08c5b7d`）。
- 熔断/服务名识别：`CircuitBreakerFilter.determineServiceName` 已同步为合后服务拓扑。

**合后服务 id 清单（网关可见）**：
- `gateway` (10081)
- `mall-catalog-service` (10082)
- `mall-user-service` (10083)
- `mall-trade-service` (10084)
- `platform-service` (10085)
- `mall-social-service` (10095)
- `factory-service` (10101)
- `ai-service` (10103)
- `scm-inventory-service` (10105)
- `scm-logistics-service` (10107)
- `finance-service` (10110)
- `blockchain-service` (10114)
- `data-pipeline` (10122)
- `erp-management-service` (10125)

## 2. GraphQL Federation（网关聚合）

网关 [`FederationRouter`](../server/gateway/src/main/java/com/metawebthree/gateway/graphql/FederationRouter.java) 定义了子图映射：

| 子图 | 服务 id | GraphQL 端点 |
|---|---|---|
| product/recommendation | `mall-catalog-service` | `http://mall-catalog-service/graphql`（含 `/graphql/product`、`/graphql/recommendation`） |
| order/cart | `mall-trade-service` | `http://mall-trade-service/graphql`（含 `/graphql/order`、`/graphql/cart`） |
| user | `mall-user-service` | `http://mall-user-service/graphql` |
| inventory | `scm-inventory-service` | `http://scm-inventory-service/graphql` |
| crm（部分） | `erp-management-service` | `http://erp-management-service/graphql` |

（其余域如 platform/factory/ai/scm-logistics/finance/blockchain/data-pipeline/mall-social 未注册 GraphQL Federation 子图，以 REST 为主。）

## 3. RPC 调用（Dubbo / Feign）

### Dubbo（ZK + Triple，gRPC 协议族）
- 服务间内部调用优先使用 Dubbo（`@DubboReference`）。网关侧的 Dubbo 客户端位于 [`server/gateway/src/main/java/com/metawebthree/gateway/client/`](../server/gateway/src/main/java/com/metawebthree/gateway/client/)：
  - `InventoryClient` → 调用 `scm-inventory-service`（通过 common 接口定义）
  - `CategoryClient`、`ProductClient` → `mall-catalog-service`
  - `OrderClient` → `mall-trade-service`
  - `UserClient` → `mall-user-service`

- CS 模块 [`platform-domain/platform-service/.../UserQueryPortImpl.java`](../server/platform-domain/platform-service/src/main/java/com/metawebthree/cs/infrastructure/client/UserQueryPortImpl.java)（HTTP fallback/default）：默认 URL `http://mall-user-service:10083`（已修复）。

### Feign（REST 声明式客户端）
- [`scm-inventory-service/.../ProcurementServiceClient.java`](../server/supply-chain-domain/scm-inventory-service/src/main/java/com/metawebthree/supplier/infrastructure/rpc/ProcurementServiceClient.java)：`@FeignClient(name = "scm-logistics-service", path = "/api/procurement")` —— 指向合并后的 `scm-logistics-service`。

> 注：绝大部分跨服务调用走 ZK + Dubbo（推荐），少量场景使用 Feign/HTTP。所有目标服务名均已更新为合后拓扑。

## 4. 前端调用映射

### 后台管理端（backstage-admin）
[`apps/backstage-admin/src/utils/http.ts`](../apps/backstage-admin/src/utils/http.ts) 的 `SERVICE_PREFIX_MAP`（按长度倒序匹配）包含：
```text
/api/forecasting, /api/v1/route-optimizer -> ai-service
/project-service -> erp-management-service
/mall-trade-service, /mall-catalog-service, /mall-user-service, /platform-service,
/factory-service, /ai-service, /scm-inventory-service, /scm-logistics-service,
/finance-service, /erp-management-service, /blockchain-service, /data-pipeline
```
匹配规则：`url.startsWith(prefix)` 时，拼接 `/{serviceId}{url}` 转发到网关（网关再按服务名路由）。

### C 端（client）
[`apps/client/src/lib/query/hooks.ts`](../apps/client/src/lib/query/hooks.ts) 等文件中的硬编码服务名已同步为合后服务：
- `promotion-service/order-service/cart-service` → `mall-trade-service`
- `product-service/recommendation-service` → `mall-catalog-service`
- `user-service` → `mall-user-service`
- `forecasting/route-optimizer` → `ai-service`（通过后台映射或直接路径处理）

## 5. 小结

- 网关 REST 路由、GraphQL Federation 子图、Dubbo/Feign 客户端、前端 baseURL 映射均已指向 **15 个合后服务**（`gateway/mall-user/mall-trade/mall-catalog/mall-social/platform/scm-inventory/scm-logistics/finance/erp-management/factory/ai/blockchain/data-pipeline`）。
- 关键跨服务引用点已核对完毕：`UserQueryPortImpl`（platform→mall-user）、`ProcurementServiceClient`（scm-inventory→scm-logistics）、网关 `client/*`（gateway→各域服务）、GraphQL 子图映射等均为合后服务名。
- Phase 6 回归（14/14 STARTED）验证了服务注册/启动拓扑一致。

> 本快照为合并前后依赖的要点梳理，不是全量调用图，但覆盖了路由/聚合/RPC/前端四大入口，足以支撑合并验证要求。
