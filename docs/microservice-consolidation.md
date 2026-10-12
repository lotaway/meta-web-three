# 微服务收敛方案 (46 → 15)

> 背景：当前 46 个 Java 微服务对本地开发/中小团队运维成本过高（46 × JVM 固定开销，
> 曾出现 21 个服务无法启动、内存吃几十 GB）。目标：按业务边界收敛到 15 个服务，
> 每个服务承担一个清晰的业务域，减少固定开销并简化部署。

## 目标架构

| 合并后服务 | 端口 | 来源服务（现状模块） | 原端口（释放） |
|---|---|---|---|
| gateway | 10081 | gateway（保留） | - |
| mall-user-service | 10083 | user-service, user-action-service, risk-control-service | 10091, 10098 |
| mall-trade-service | 10084 | order-service, payment-service, cart-service, promotion-service, after-sale-service | 10086, 10089, 10090, 10093 |
| mall-catalog-service | 10082 | product-service, mall-supplier-service, recommendation-service, review-service | 10096, 10104, 10097 |
| mall-social-service | 10095 | live-service, group-buying-service, social-commerce-service | 10094, 10100 |
| platform-service | 10085 | message-service, media-service, commission-service, cs-service, data-analysis-service, tenant-service, developer-portal-service | 10087, 10088, 10092, 10099, 10126, 10115 |
| scm-inventory-service | 10105 | inventory-service, warehouse-service, supplier-service, dom-service, inventory-alert-service | 10106, 10109, 10124, 10121 |
| scm-logistics-service | 10107 | logistics-service, procurement-service, rma-service | 10108, 10123 |
| erp-finance-service | 10110 | finance-service, invoice-service, reporting-service, settlement-service | 10111, 10112, 10113 |
| erp-management-service | 10125 | crm-service, hrm-service, project-service | 10119, 10120 |
| factory-service | 10101 | mes-service, digital-twin-service, production-service | 10102, 10116 |
| ai-service | 10103 | forecasting-service, route-optimizer | 10117 |
| blockchain-service | 10114 | wallet-service, traceability-service | 10118 |
| data-pipeline | 10122 | data-pipeline（保留） | - |

> 注：ai-domain 下还有 `ai-warehouse-service`、`risk-scorer` 两个模块未在任何 registry 中注册，本次收敛不处理。

## 每个合并的标准步骤

1. **前置检查**：确认被合并服务有没有外部按名/按端口引用（`grep <name>|端口`），记录调用方。
2. **代码搬移**：把被合并服务的 `src/main/java/<pkg>` 整体拷入目标服务模块，保留原包名（避免类级修改）。
3. **组件扫描**：目标服务 `@SpringBootApplication(scanBasePackages = { "<目标包>", "<被合并包>" })`。
   - 避免用 `com.metawebthree` 全包扫描（会连带加载 common.audit 的 AOP/Service，导致 Mapper 缺失类问题）。
4. **MyBatis 扫描**：目标服务加 `@MapperScan(basePackages = { <所有业务包> }, annotationClass = Mapper.class)`，
   保证所有 `@Mapper` 接口（含被合并服务的）都被注册，且不误扫普通接口。
5. **资源合并**：`db/schema.sql` 追加；`application.yml` 不合并（统一用目标服务端口/应用名），
   删除被合并服务的独立 yml。
6. **移除模块**：从 `server/pom.xml` `<modules>` 与 `scripts/server-services-registry.sh` 删除；
   删除旧模块目录（git 可回滚）。
7. **gateway/客户端核对**：gateway 路由、前端 baseURL、跨服务 RPC 更新（如有按端口/服务名引用）。
8. **验证**：`mvn clean install` 目标模块 → 用 `verify-all.sh` 单服务启动检查 → 访问各业务端点冒烟。

## 阶段划分

- **Phase 0** 基线与映射（本文件建立；46 服务盘点）
- **Phase 1** ERP：`invoice/reporting/settlement → finance-service`；`hrm/crm/project → erp-management-service` —— ✅ 已完成（含构建 + 启动验证）
- **Phase 2** 供应链：`inventory/warehouse/supplier/dom/inventory-alert → scm-inventory-service`；`logistics/procurement/rma → scm-logistics-service` —— ✅ 已完成（含构建 + 启动验证）
- **Phase 3** 电商 mall：`product/mall-supplier/recommendation/review → mall-catalog-service`（端口 10082，双 GraphQL 子图 /graphql/product、/graphql/recommendation）✅；`order/payment/cart/promotion/after-sale → mall-trade-service`（端口 10084，双 GraphQL 子图 /graphql/order、/graphql/cart）✅；`user/user-action/risk-control → mall-user-service`（端口 10083，单 GraphQL 子图 /graphql）✅；`live/group-buying/social-commerce → mall-social-service`（端口 10095，REST 双控制器 + admin 接口，无 GraphQL 子图）✅ —— 均含构建 + 启动验证，网关子图地址已切换；Phase 3 全部完成（mall 领域 13→4）
- **Phase 4** 平台：`message/media/commission/cs/data-analysis/tenant/developer-portal → platform-service`（端口 10085，REST 多控制器 + cs 会话 WebSocket + message 邮件 + media 对象存储，无 GraphQL 子图）✅ —— 构建 + 启动验证通过（63s），各模块端点冒烟可达（含 common 配置补装：Security/Database/Redis/RestClient/Swagger/Web/SecurityInterceptor/Minio/S3/MQProducer/DistributedCacheService/GlobalExceptionHandler）
- **Phase 5** 工厂 / AI / 区块链：`forecasting/route-optimizer → ai-service`（端口 10103）✅；`wallet/traceability → blockchain-service`（端口 10114）✅ —— 均含构建 + 启动验证（业务端点 500 属 dev 库缺表，非接线问题）；`mes/digital-twin/production → factory-service`（端口 10101）⚠️ 脚手架 + 构建通过，Dockerfile/compose/端口已接线，但单独启动受阻：mes 领域层有 4 个 `@Service` 依赖 7 个「只有接口、无实现、无数据表」的仓储（andon / equipment-status / equipment-type / work-order-split 半成品），详见 TODO.md Phase 5
- **Phase 6** 收尾：`run-server.sh` 全量回归、内存复测、文档/网关/前端同步

## 验证与回滚

- 每合并一个就：构建 + 单服务启动 + 冒烟，通过后才合并下一个。
- 所有改动在 git 中可回滚（旧模块目录删除前先确保已搬移完整）。
- 完成后跑 `bash scripts/check-server-memory.sh` 对比合并前后内存。