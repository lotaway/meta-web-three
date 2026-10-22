# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRINCIPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

### 1. apps/backstage-admin 环境参数与默认值 — [x]
- `.env.example` 补齐全部必须/可选环境参数：`VITE_BASE_SERVER_URL`（示例 `http://192.168.1.110:10081`）、`VITE_API_DOC_HOST`、`VITE_UPLOAD_URL`（默认 `/platform-service/minio/upload`）、`VITE_BLOCKCHAIN_ENABLED`
- 新增 `src/utils/env.ts` 集中读取环境变量并兜底默认值（`VITE_BASE_SERVER_URL` 默认 `http://localhost:10081`）
- `http.ts` / `singleUpload.vue` / `multiUpload.vue` / `Tinymce/index.vue` / `router/guard.ts` / `stores/permission.ts` 全部改为引用该模块
- **效果**: axios baseURL 不再可能为空，避免全部 API 请求失败
- **验证**: `npm run type-check` EXIT=0，`npm run build` 成功（45.48s）

### 2. SERVICE_PREFIX_MAP 新增 /api/admin/user — [x]
- 新增 `'/api/admin/user': 'mall-user-service'`
- 前缀按长度倒序匹配，`/api/admin/user-action` 不受影响
- **验证**: 网关实测 `/mall-user-service/api/admin/user/{id}` 命中 mall-user-service

### 3. /prefrenceArea 路由修正 — [x]
- 服务名由 `mall-catalog-service` 改为 `mall-trade-service`
- 依据：前端类型 `CmsPrefrenceArea` 对应 mall-trade-service 的 `CmsPrefrenceAreaDO`（含 `listAll`）
- **验证**: 网关实测 `/mall-trade-service/prefrenceArea/listAll` 命中 mall-trade-service

### 4. /productCategory 统一为 /product-category — [x]
- `http.ts` 前缀改为 `'/product-category': 'mall-catalog-service'`
- 后端 `ProductCategoryAdminController` 的 `@RequestMapping` 同步由 `/productCategory` 改为 `/product-category`，与 `ProductCategoryController` 对齐
- **验证**: 网关实测 `/mall-catalog-service/product-category/list/1` 命中 mall-catalog-service

### 5. productCate.ts 及调用处 URL 同步 — [x]
- 8 个接口由 `/productCategory/...` 改为 `/product-category/...`
- 全仓库已无 `/productCategory` 路径引用

### 6. apps/client 手写 API 的 service 前缀与端口核对 — [x]
- service 前缀与网关 discovery locator 约定（`/{serviceId}/**`）一致
- 端口统一 10081（`src/lib/api/baseUrl.ts` 的 `DEFAULT_API_PORT`），并修复空字符串环境变量会遮蔽备用地址的缺陷（`??` 不把 `''` 当未设置）
- 同时修正 `src/lib/query/hooks.ts` 中 8 处与后端实际路径不符的子路径：`/products`→`/product`、`/categories`→`/product-category/children`、`/brands`→`/brand/list`、`/orders`→`/order/list`、`/cart/items`→`/cart/list`、购物车删除改 `POST /cart/delete`、数量更新改 `GET /cart/update/quantity`、`/user/profile`→`/user/info`
- **验证**: `npx tsc --noEmit` EXIT=0

### 7. apps/client 运行 openapi-generator — [x]
- 网关聚合 OpenAPI 已按 `/{serviceId}/**` 加服务前缀（`AggregatedApiDocsController` 修复）
- 生成客户端 basePath 带 serviceId：`/mall-catalog-service/product-category/create`、`/mall-user-service/api/admin/user/{id}`、`/mall-trade-service/order/{id}/refund`、`/mall-trade-service/prefrenceArea/listAll` 等
- 生成客户端 baseUrl 指向网关 10081：`BASE_PATH = \`${process.env.NEXT_PUBLIC_BACK_API_HOST || "http://192.168.1.110:10081"}\``（环境变量优先，代码内不写死）
- `apps/tools/OpenApiToTS.js` 的 runtime 回退地址改用 API host 而非文档 host（文档 host 可能与业务端口不同）；`generate-api.js` 改为加载 `.env` 而非仅 `.env.development`
- **验证**: 生成 137+140+182 条带前缀路径，client `tsc --noEmit` EXIT=0

### 8. 联调验证 — [x]
- 新增 `scripts/verify-gateway-routing.sh`（用各服务自带 `/v3/api-docs` 作探针，该关键字在网关 excluded-path-keywords 中，能跳过认证直接进入路由，返回该服务自己的 OpenAPI 文档即证明命中）
- **结果**: 全部 PASS
  - `mall-catalog-service` / `mall-user-service` / `mall-trade-service` 三个服务的请求均路由到各自服务
  - `/mall-trade-service/flash/current`、`/mall-user-service/user/signIn` 等业务路径可达目标服务
  - 聚合 OpenAPI 中三个服务的路径全部带 serviceId 前缀

## 配套改进（支撑上述任务）
- [x] `scripts/server-services-registry.sh` 新增 `SERVER_PORT_OFFSET`（默认 0）；`run-mall-server.sh` / `run-server.sh` 以注册表端口启动服务。Windows 的 Hyper-V/WSL 会预订 100 个端口一段的 excluded port range，默认端口段 10081-10125 被预订时服务无法绑定（报 `Port 10081 was already in use` / WSAEACCES），用偏移即可整栈挪到空闲端口段
- [x] `server/gateway/.../AggregatedApiDocsController` 合并各服务 OpenAPI 路径时补 `/{serviceId}` 前缀，与网关 discovery locator 路由约定一致（同时消除不同服务同路径被覆盖的问题）
- [x] `scripts/server-services-registry.sh` 新增 `server_repair_corrupted_targets`：检测 Eclipse/JDT 编译产生的 `Unresolved compilation problem` class 与半成品 boot jar 并清理对应模块 target，避免 Maven 报 "Nothing to compile" 把损坏 class 打进 jar 导致服务启动即挂
- [x] `.vscode/settings.json` 增加 `java.autobuild.enabled: disabled`，让 Maven 成为 `target/classes` 的唯一产出方
- [x] 文档同步：`apps/backstage-admin/README.md`（环境变量表 + 网关路由约定）、`apps/client/README.md`（生成流程 + serviceId 前缀 + 端口 10081）、`apps/client/.env.example`、`server/README.md`（端口偏移 + 集成测试外部依赖）

## 验证汇总
| 项目 | 结果 |
|------|------|
| 后端全仓库编译 (`mvn -o clean compile`) | EXIT=0 |
| 后端全量测试 (`mvn -o clean test`) | BUILD SUCCESS，269 用例 0 失败 0 错误（8 skipped = 需 PostgreSQL 的 MesPostgresTest） |
| backstage-admin `npm run type-check` | EXIT=0 |
| backstage-admin `npm run build` | 成功（45.48s） |
| client `npx tsc --noEmit` | EXIT=0 |
| openapi-generator 生成 | 成功，路径带 serviceId，baseUrl 指向网关 10081 |
| 网关路由联调 `scripts/verify-gateway-routing.sh` | 全部 PASS |

## 遗留说明
- client `npx jest` 7 个 suite 启动失败（`TypeError: Object.defineProperty called on non-object`），stash 改动后同样失败，为既有的 jest 环境问题，非本次引入
- `apps/tools/DownloadGraphQLSchema.js` 在 tools 目录缺少 `graphql` 模块依赖，GraphQL schema 下载失败；codegen 使用既有 `graphql/schema.graphql` 正常生成，与本次 REST 客户端生成无关
