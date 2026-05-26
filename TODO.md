# TODO

## P3: 前端组件代码质量修复

### DemandChart.tsx — 子组件超行

- [ ] 目标是将 `ChartStats`（当前 52 行）拆分至 ≤20 行
- [ ] 目标是将 `ChartLegend`（当前 35 行）拆分至 ≤20 行
- [ ] 目标是将 `YAxisLabels`（当前 31 行）拆分至 ≤20 行
- [ ] 目标是将 `DataPoints`（当前 36 行）拆分至 ≤20 行
- [ ] 目标是将 `XAxisLabels`（当前 30 行）拆分至 ≤20 行
- [ ] 目标是检查并拆分同文件中其他超行子组件（ChartHeader、GridLines、ConfidenceInterval、ForecastPath、ActualPath）
- 难点：子组件内部有内联样式、计算逻辑和渲染标记混在一起，需逐一定义 Props 接口再独立文件提取

### InventoryAlertPanel.tsx — 超 500 行

- [ ] 目标是将 InventoryAlertPanel.tsx（当前 506 行）缩减至 ≤500 行
- [ ] 目标是将 506 行对应的逻辑子组件拆分到独立文件（如 AlertListItem、AlertFilterBar 等）
- 难点：仅超 6 行，需识别内聚性子组件边界，最小的拆分就能达标


## P2: WarehouseApplicationServiceImpl 重构

路径：`server/supply-chain-domain/warehouse-service/src/main/java/com/metawebthree/warehouse/application/WarehouseApplicationServiceImpl.java`

### CQRS 命令查询职责分离

- [ ] 目标是让 `createWarehouse`（第 47 行）返回 `void` 而非 `WarehouseDTO`
- [ ] 目标是让 `updateWarehouse`（第 71 行）返回 `void` 而非 `WarehouseDTO`
- [ ] 目标是让 `createInboundOrder`（第 106 行）返回 `void` 而非 `InboundOrderDTO`
- [ ] 目标是让 `confirmInboundOrder`（第 143 行）返回 `void` 而非 `InboundOrderDTO`
- [ ] 目标是让 `completeInboundOrder`（第 155 行）返回 `void` 而非 `InboundOrderDTO`
- 难点：Controller 层目前依赖这些方法返回 DTO，需同步修改 Controller 及调用方；可能涉及前端 API 响应结构调整

### Optional.orElse(null) 替换

- [ ] 目标是替换第 90 行 `updateWarehouse` 中的 `.orElse(null)` 为抛业务异常
- [ ] 目标是替换第 97 行 `queryWarehouse` 中的 `.orElse(null)` 为抛业务异常
- [ ] 目标是替换第 151 行 `confirmInboundOrder` 中的 `.orElse(null)` 为抛业务异常
- [ ] 目标是替换第 180 行 `completeInboundOrder` 中的 `.orElse(null)` 为抛业务异常
- [ ] 目标是替换第 187 行 `queryInboundOrder` 中的 `.orElse(null)` 为抛业务异常
- 原则：查询不到应抛 `WarehouseNotFoundException` / `InboundOrderNotFoundException`，由全局异常处理器统一返回 404
- 难点：需先确认这些异常类和全局处理器是否存在，不存在则新建

## 测试补充

### digital-twin-api.test.ts

- [ ] 目标是给 `fetchDevices` 补充 HTTP 调用 Mock 和响应验证（当前仅验证函数存在）
- [ ] 目标是给 `fetchActiveAlerts` 补充 HTTP 调用 Mock 和响应验证
- [ ] 目标是给 `fetchStatsSummary` 补充 HTTP 调用 Mock 和响应验证
- 路径：`apps/digital-twin/system-management/src/renderer/services/digital-twin-api.test.ts`
- 难点：Mock axios 的网络请求，验证请求 URL、请求方法、响应数据结构的完整性

### 端到端测试

- [ ] 目标是实现 3D 场景加载的 E2E 测试，验证场景初始化、模型加载、相机控制功能
- [ ] 目标是实现实时数据展示的 E2E 测试，验证 WebSocket 推送到 UI 更新链路
- [ ] 目标是实现告警流程的 E2E 测试，覆盖告警创建 → 推送 → 前端展示 → 告警确认/关闭全链路
- 难点：E2E 测试框架待确认（Playwright/Cypress），3D 场景测试需处理 WebGL 渲染和动画帧

## 跨服务鉴权体系延伸

### ERP 各服务接入 Gateway

- [ ] 目标是将 ERP 各服务注册到 ZooKeeper，实现 Gateway 自动路由
- [ ] 目标是 ERP 服务消费 Gateway 传递的 `X-User-Id` / `X-User-Role` 请求头
- [ ] 目标是 ERP 服务使用 `@RequirePermission` 注解进行接口鉴权
- [ ] 目标是统一规划 ERP 权限资源树，避免自建鉴权轮子

### 供应链各服务接入 Gateway

- [ ] 目标是将供应链各服务注册到 ZooKeeper，实现 Gateway 自动路由
- [ ] 目标是供应链服务消费 Gateway 传递的 `X-User-Id` / `X-User-Role` 请求头
- [ ] 目标是供应链服务使用 `@RequirePermission` 注解进行接口鉴权
- [ ] 目标是统一规划供应链权限资源树，避免自建鉴权轮子
