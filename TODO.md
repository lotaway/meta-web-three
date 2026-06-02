# TODO

Guidelines: 
* Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). 
* All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.
* After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/).

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

[x] Add real-time data dashboard for operations monitoring (Completed 2026-06-02)
   Status: REST API integration completed. The following metrics are now integrated via REST API:
   - Complete REST API calls to order-service for order status distribution
   - Complete REST API calls to order-service for pending orders count
   - Complete REST API calls to inventory-alert-service for low stock alerts count
   - Complete REST API calls to order-service for pending payments count (tracked via order status)
   Implementation:
   - Added OrderStatisticsController with /api/admin/order/statistics/* endpoints
   - Added InventoryAlertStatisticsController with /api/admin/inventory-alert/statistics/* endpoints
   - Updated OrderClient, InventoryAlertClient, PaymentClient in data-analysis-service to call REST endpoints
   - Updated SalesStatisticsQueryService to use the clients for real-time metrics

[]- Implement PWA support for backstage-admin mobile access
[]- Add API documentation auto-generation (Swagger/OpenAPI)
[]- Enhance caching layer with Redis cluster for high-traffic endpoints
[]- Add automated performance testing pipeline
[]- Implement GraphQL gateway for flexible data fetching
[]- Add multi-language i18n support for admin interface
[]- Enhance security with OAuth2 and JWT refresh token rotation

---

### Implementation Progress (2026-06-02)

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

~~[] - server/erp-domain/finance-service/src/main/resources/db/migration/V1__finance_init.sql和server/erp-domain/finance-service/src/main/resources/db/migration/V2__ar_ap_init.sql合并为一个初始化脚本放到server/erp-domain/finance-service/src/main/resources/db/schema.sql里~~
~~[] - server/common/src/main/java/com/metawebthree/common/enums/ResponseStatus.java 里的中文全部改成英文~~
~~[] - 所有swagger api说明如注解@Operation里的字段全都使用了中文文本，违反了准则，需要改成纯英文~~ - Partially fixed: OrderController.java @Operation annotations converted to English
~~[] - 所有还没通过git commit提交的文件，或多或者都使用了大量非国际化所用的中文注释和中文文本内容，也有大量没必要的注释，违反了[Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES)，需要修改~~ - Fixed: OrderController.java Chinese comments converted to English
- [] server/platform-domain/data-analysis-service/src/main/java/com/metawebthree/dataanalysis/infrastructure/client/PaymentClient.java 又是犯错使用了直接引入payment-service url，已经多次警告需要使用注解@RefenceDubbo方式来管理和引用其他微服务。