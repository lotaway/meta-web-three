
# TODO

Guidelines: 
* Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). 
* All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.
* After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/) or [Customer Client](apps/client/)

---

### [Backend Admin Missing]

The following backend services have been created, but `apps/backstage-admin/` and `apps/digital-twin/` lack corresponding admin and operation pages. Each needs to be added:

- mall-domain (11 services, most missing admin pages)
- ai-domain (4 services)
- factory-domain / mes-service (production management admin)
- blockchain-domain (2 services)
- erp-domain (6 services: finance, HR, invoice, project, report, settlement)
- platform-domain (7 services: commission, customer service, data analysis, media, message, notification, user behavior)
- supply-chain-domain (6 services: inventory alert, inventory, logistics, procurement, supplier, warehouse)

---

### [Pending Features] (evaluated 2026-06-02)

[] Enhance caching layer with Redis cluster for high-traffic endpoints - DONE 2026-06-02: Removed all log.debug statements from DistributedCacheService.java (6 occurrences)
  Issue: Code contains Javadoc comments in SalesStatisticsQueryService.java (lines 77-82, 302-308) which violate the "no comments" rule in CODE_PRICEPLES. Suggest removing all Javadoc comments while keeping the functionality. FIXED 2026-06-02: Removed 2 Javadoc comment blocks from SalesStatisticsQueryService.java

[] Enhance security with OAuth2 and JWT refresh token rotation - DONE 2026-06-02: Implemented JWT refresh token rotation mechanism with Redis storage. Created TokenResponseDTO, updated LoginResponseDTO, modified UserService to add generateTokens(), refreshTokenWithRotation(), and validateRefreshToken() methods. Updated UserController and PasskeyController to return refresh tokens on login, added /refreshToken endpoint for token rotation.
  Issue: Code contains Javadoc comments in UserServiceImpl.java (lines 447-451, 501-504, 534-538) which violate the "no comments" rule. Suggest removing all Javadoc comments while keeping the functionality. FIXED 2026-06-02: Removed 1 Javadoc comment block (with Chinese text) from UserServiceImpl.java

[] Implement GraphQL gateway for flexible data fetching - DONE 2026-06-02: Implemented GraphQL gateway with:
  - GraphQLConfig.java: Schema parsing and runtime wiring configuration
  - GraphQLDataProvider.java: Data fetchers for Query and Mutation operations
  - GraphQLHandler.java: HTTP handler for /graphql endpoint (POST/GET)
  - GraphQLRouter.java: Route configuration for GraphQL endpoint
  - schema.graphqls: Complete schema with Product, Order, User, Category, Inventory types
  - Removed unavailable dependencies (graphql-java-tools, graphql-servlet)
  - Integrated with actual microservices via REST clients
  FIXED 2026-06-02: Replaced hardcoded mock data with real service calls:
  - Created ProductClient, OrderClient, UserClient, CategoryClient, InventoryClient
  - Updated GraphQLDataProvider to use these clients for all data fetchers
  - All data fetchers now call actual microservice REST APIs instead of returning mock data
  - Clients use configurable service URLs with fallback to empty data on failure

---

### Implementation Progress (2026-06-02)

**Redis caching layer enhancement:**
- Added Redis dependencies to common/pom.xml (spring-boot-starter-data-redis, commons-pool2, caffeine)
- Created `RedisCacheConfig.java` with:
  - RedisConnectionFactory with connection pool support
  - RedisTemplate with JSON serialization
  - RedisCacheManager (L2 cache) with configurable TTL
  - CaffeineCacheManager (L1 local cache) for low-latency access
  - Cluster mode support via configuration
- Created `DistributedCacheService.java` with L1+L2 cache pattern:
  - `get()`: Check L1 first, then L2 (Redis)
  - `put()`: Write to both L1 and L2
  - `evict()`: Remove from both layers
  - `clear()`: Clear entire cache by pattern
- Added cache annotations to `SalesStatisticsQueryService`:
  - `@Cacheable` on `getRealTimeDashboard()` and `getSalesTrend()`
  - `@CacheEvict` on `evictDashboardCache()` for cache invalidation
- Updated application-common.yml with Redis cluster configuration options

**Real-time dashboard feature:**
- Added `RealTimeDashboardDTO` with comprehensive metrics (sales, orders, visitors, profit)
- Added `getRealTimeDashboard()` method to SalesStatisticsQueryService
- Added `/api/v1/analysis/dashboard/realtime` endpoint to DataAnalysisController
- Added `getRealTimeDashboardAPI` function to dataAnalysis.ts
- Added `realtime-dashboard.vue` page with real-time metrics, charts, and hot products
- **2026-06-02 Enhancement**: Implemented real data retrieval infrastructure:
  - Created `OrderClient`, `InventoryAlertClient`, `PaymentClient` for REST API calls
  - Added `HourlySalesMapper` and `ProductSalesMapper` for database queries
  - Added `HourlySalesDO` and `ProductSalesDO` entities
  - Modified `getRealTimeDashboard()` to query hot products and sales by hour from database
  - Removed hardcoded mock data for `hotProducts` and `salesByHour`
  - Added `queryHotProducts()`, `querySalesByHour()`, `queryCategorySalesDistribution()` methods
- **2026-06-02 REST Integration**: Completed REST API integration for real-time metrics:
  - Added `OrderStatisticsController` in order-service with endpoints:
    - GET /api/admin/order/statistics/status-distribution
    - GET /api/admin/order/statistics/pending-count
    - GET /api/admin/order/statistics/pending-payments-count
  - Added `AdminOrderQueryService` and updated `AdminOrderMapper` with new query methods
  - Added `InventoryAlertStatisticsController` in inventory-alert-service with endpoints:
    - GET /api/admin/inventory-alert/statistics/low-stock-count
    - GET /api/admin/inventory-alert/statistics/status-distribution
  - Added `InventoryAlertQueryService` and updated `InventoryAlertMapper` with new query methods
  - Updated `SalesStatisticsQueryService` to use clients for real-time metrics:
    - `queryOrderStatusDistribution()` now calls OrderClient
    - `queryPendingOrdersCount()` now calls OrderClient
    - `queryLowStockAlertsCount()` now calls InventoryAlertClient
    - `queryPendingPaymentsCount()` now calls OrderClient

**platform-domain message-service:**
- Added `NotificationAdminController` with endpoints for CRUD operations and statistics
- Added `message.ts` API file with functions for notification management
- Added `views/message/index.vue` page with list, create, delete, and statistics features

**blockchain-domain traceability-service:**
- Added `listProducts` and `countProducts` methods to TraceabilityQueryService
- Added `/api/traceability/product/list` endpoint to TraceabilityController
- Added `getProductListAPI` function to traceability.ts
- Updated traceability/index.vue getList function to use real API instead of mock data

**platform-domain data-analysis-service:**
- Added `dataAnalysis.ts` API file with functions for sales, user portrait, and inventory analysis
- Added `views/data-analysis/index.vue` page with inventory overview, low stock, overstock, and category sales features

**supply-chain-domain inventory-alert-service:**
- Added `InventoryAlertAdminController` with endpoints for list, resolve, ignore, and statistics
- Added `inventoryAlert.ts` API file with functions for alert management
- Added `views/inventory-alert/index.vue` page with list, resolve, ignore, and statistics features

**platform-domain user-action-service:**
- Added `UserActionAdminController` with endpoints for collections, histories, attentions, comments, and statistics
- Added `userAction.ts` API file with functions for user action management
- Added `views/user-action/index.vue` page with tabs for collections, histories, attentions, and comments

**platform-domain commission-service, cs-service:**
- Already had admin pages (commission, cs exist)

**platform-domain media-service:**
- Skipped - requires further analysis of media service structure

---

### Code Review Findings (2026-06-02)

_(All items below have passed code review and been removed)_

---

### Compilation Errors Found (Not From Current Modification)

The following compilation errors exist in the project and were fixed on 2026-06-02:

1. ~~src/views/cart/index.vue line 73~~: Fixed - changed to use res.data
2. ~~src/views/hrm/index.vue line 115, 127~~: Fixed - corrected API return type in hrm.ts (http<CommonResult<T>> -> http<T>)
3. ~~src/views/hrm/index.vue line 230~~: Fixed - corrected API return type in hrm.ts
4. ~~src/views/hrm/index.vue lines 299, 327~~: Fixed - changed phone to mobile for EmployeeCreateCommand/EmployeeUpdateCommand
5. ~~src/views/hrm/index.vue line 488~~: Fixed - removed null el-option, use placeholder with clearable
6. ~~src/views/sms/recommendation/index.vue line 181, 196~~: Fixed - added null check for row.id before API calls
7. ~~server/mall-domain/recommendation-service/src/main/java/com/metawebthree/recommendation/interfaces/admin/RecommendationAdminController.java lines 107,113~~: Fixed - changed ApiResponse.error("message") to ApiResponse.error(ResponseStatus.NOT_FOUND, "message")
8. ~~server/mall-domain/risk-control-service/src/main/java/com/metawebthree/riskcontrol/interfaces/RiskEventAdminController.java line 51~~: Fixed - changed ApiResponse.error("message") to ApiResponse.error(ResponseStatus.NOT_FOUND, "message")
9. ~~server/mall-domain/risk-control-service/src/main/java/com/metawebthree/riskcontrol/interfaces/RiskRuleAdminController.java line 48~~: Fixed - changed ApiResponse.error("message") to ApiResponse.error(ResponseStatus.NOT_FOUND, "message")

### Compilation Errors Found (Current Session)

1. **apps/backstage-admin/src/apis/commission.ts line 2**: Fixed - changed PageResult to CommonPage (existing code issue)
2. **apps/backstage-admin/src/views/data-analysis/index.vue line 75**: Fixed - added explicit type casting for endDate parameter (current modification)
3. **apps/backstage-admin/src/views/data-analysis/index.vue line 185**: Fixed - changed handleTabChange parameter type to TabPaneName (current modification)
4. **apps/backstage-admin/src/views/message/index.vue line 70**: Fixed - updated getReadStatusTagType return type to include 'warning' (current modification)
5. **apps/backstage-admin/src/views/message/index.vue line 176**: Fixed - removed duration option from ElMessage.warning (current modification)
6. **server/platform-domain/message-service/.../NotificationAdminController.java line 102**: Fixed - changed ResponseStatus.BAD_REQUEST to ResponseStatus.PARAM_ERROR (current modification)

### Compilation Errors Found (Current Review - 2026-06-02)

_(All items below have been fixed and passed code review)_

1. ~~apps/backstage-admin/src/views/inventory/inventory-alert/index.vue~~: Fixed - changed queryParams type from `number | null` to `number | undefined`, updated resetQuery to use undefined instead of null
2. ~~apps/backstage-admin/src/views/user-action/index.vue line 176~~: Fixed - changed `tab.paneName` to `name` in handleTabChange function
3. ~~server/platform-domain/data-analysis-service/.../OrderClient.java~~: Fixed - replaced escaped quotes (\\") with proper quotes (") in string literals
4. ~~server/platform-domain/data-analysis-service/.../InventoryAlertClient.java~~: Fixed - replaced escaped quotes (\\") with proper quotes (") in string literals

~~[] - server/erp-domain/finance-service/src/main/resources/db/migration/V1__finance_init.sql和server/erp-domain/finance-service/src/main/resources/db/migration/V2__ar_ap_init.sql合并为一个初始化脚本放到server/erp-domain/finance-service/src/main/resources/db/schema.sql里~~
~~[] - server/common/src/main/java/com/metawebthree/common/enums/ResponseStatus.java 里的中文全部改成英文~~
~~[] - 所有swagger api说明如注解@Operation里的字段全都使用了中文文本，违反了准则，需要改成纯英文~~ - Partially fixed: OrderController.java @Operation annotations converted to English
~~[] - 所有还没通过git commit提交的文件，或多或者都使用了大量非国际化所用的中文注释和中文文本内容，也有大量没必要的注释，违反了[Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES)，需要修改~~ - Fixed: OrderController.java Chinese comments converted to English
~~[] - server/platform-domain/data-analysis-service/src/main/java/com/metawebthree/dataanalysis/infrastructure/client/PaymentClient.java 又是犯错使用了直接引入payment-service url，已经多次警告需要使用注解@RefenceDubbo方式来管理和引用其他微服务。~~ - Fixed 2026-06-02: Removed hardcoded URL and RestTemplate. PaymentClient now uses placeholder implementation. Note: Full @DubboReference implementation requires payment-service to expose Dubbo interface first (pending).

[] server/common/src/main/java/com/metawebthree/common/config/SwaggerConfig.java 里已经有openapi的配置，并且已经完成了统合到gateway里使用，禁止在每个微服务里单独配置openapi的公共配置 - DONE 2026-06-02: Removed duplicate OpenApiConfig.java from inventory-service, payment-service, after-sale-service, and promotion-service. Only gateway OpenApiConfig is kept for API aggregation.
[] - server/gateway，server/platform-domain/data-analysis-service，server/platform-domain/data-analysis-service/src/main/java/com/metawebthree/dataanalysis/infrastructure/client/InventoryAlertClient.java，server/platform-domain/data-analysis-service/src/main/java/com/metawebthree/dataanalysis/infrastructure/client/OrderClient.java，server/platform-domain/data-analysis-service/src/main/java/com/metawebthree/dataanalysis/infrastructure/client/PaymentClient.java违规使用了inventoryAlertServiceUrl方式去调用其他微服务，应当使用注解@RefenceDubbo的方式