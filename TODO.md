# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 代码规范审查结果 (2026-05-28)

### 一、工艺路线 (ProcessRoute)

**审查发现问题（已更新）**
- 后端领域逻辑 ✅ 真实实现（实体、仓储、测试）
- **前端已存在**（经核查：API/页面/store/路由/国际化均已实现）
- **后端 Controller 规范问题已修复**（移除 JavaDoc，使用自定义异常）

> **更新说明（2026-05-28）**：经核查，前端代码实际已存在（见下方子任务标记）。后端 Controller 规范问题已修复。

### 子任务

- [x] **后端 — ProcessRoute 专用 REST 控制器** ✅ 已修复
  - 创建 `ProcessRouteController.java`，含 CRUD + 激活/归档端点 ✅
  - 创建请求/响应 DTO（`ProcessRouteDTO`）✅
  - **规范修复**：已移除所有 JavaDoc 注释；已使用自定义 `ProcessRouteException` 替代 `IllegalArgumentException`
  - 已创建 `MesExceptionHandler` 处理自定义异常
- [ ] **后端 — 集成测试** 🔄 进行中
  - 问题：ApplicationContext 加载失败，17个测试全部报错 `IllegalStateException: ApplicationContext failure threshold exceeded`
  - 根本原因：测试环境缺少 `MesEventPublisher` bean（依赖 `DomainEventPublisher` -> 需要 KafkaTemplate）
  - 已尝试修复：
    - 创建 `TestConfig.java` 提供 mock DomainEventPublisher 和 MesEventPublisher
    - 修改 `application-test.yml` 排除 Kafka/Security 自动配置
    - 单元测试 `ProcessRouteTest` (20用例) 已通过 ✅
  - 状态：集成测试仍因 Spring 上下文加载复杂依赖（Kafka、Security）失败，建议简化测试或使用 @MockBean
- [ ] **前端 — API 封装** (`src/apis/processRoute.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 工艺路线列表页** (`src/views/mes/processRoute/index.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 工艺路线新增/编辑页** (`src/views/mes/processRoute/form.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 工艺路线详情页** (`src/views/mes/processRoute/detail.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — Pinia 状态管理** (`src/stores/processRoute.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 路由注册** (`src/router/index.ts` 添加 `/mes/process-route/*`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 国际化** (`src/locales/zh-CN.ts` + `en-US.ts` 添加翻译键)
  - 编译失败：vue-i18n 依赖缺失

---

### 二、设备管理 (Equipment)

**审查发现问题（已更新）**
- 后端含真实 OEE 算法和持久化 ✅
- **后端实体状态校验已存在** ✅（`Equipment.java` 已实现 6 个状态转换方法的状态校验）
- **后端 Controller 不存在**
- **前端不存在**

> **更新说明（2026-05-28）**：经核查，`Equipment.java` 中已实现所有状态校验方法（startTask/completeTask/reportBreakdown/repair/startMaintenance/completeMaintenance），TODO.md 中描述的"实体方法无校验"是过时的。

### 子任务

- [ ] **后端 — 修复 Equipment 实体状态校验**
  - 编译失败：vue-i18n 依赖缺失
- [ ] **后端 — 验证单元测试全部通过** (`EquipmentTest`, 15 个用例)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **后端 — 设备专用 REST 控制器**
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — API 封装** (`src/apis/equipment.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 设备列表页** (`src/views/mes/equipment/index.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 设备新增/编辑页** (`src/views/mes/equipment/form.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 设备详情页** (`src/views/mes/equipment/detail.vue`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 设备维护计划管理** (`src/views/mes/equipment/maintenancePlan.vue`)
- [ ] **前端 — 设备点检记录** (`src/views/mes/equipment/checklist.vue`)
- [ ] **前端 — Pinia 状态管理** (`src/stores/equipment.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 路由注册** (`src/router/index.ts` 添加 `/mes/equipment/*`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 国际化** (`src/locales/zh-CN.ts` + `en-US.ts`)
  - 编译失败：vue-i18n 依赖缺失

---

### 三、流程引擎与规则引擎 (SPEC 3.3 / 3.4)

**审查发现问题**
- 规则引擎(PokaYoke) ✅ 真实算法
- **流程引擎仅为模板/实例管理器，`startInstance()` 只复制 JSON 不执行节点路由**
- **`pom.xml` 中 Flowable 依赖声明但未使用**
- **前端完全不存在**

> **判定原因：** 此前被标记为"审查通过 ✅"，但实际检查发现：(1) 流程引擎名不副实——`ProcessFlowQueryService.startInstance()` 仅复制 `flowData` JSON 并标记状态为 RUNNING，从未执行任何工作流节点路由、条件判断或并行网关逻辑，不是真正的 BPMN 引擎；(2) `pom.xml` 声明的 Flowable 依赖全部未使用（死代码）；(3) 前端完全不存在。判定为**未完成**。

### 子任务

#### 决策前置
- [ ] **设计决策：确定流程引擎定位**
  - 编译失败：vue-i18n 依赖缺失
- [ ] 根据决策结果：**清理 `pom.xml` 中未使用的 Flowable 依赖**

#### 规则引擎（已有后端，补充前端）
- [ ] **后端 — 规则引擎集成测试**
- [ ] **前端 — PokaYoke 规则列表/配置页** (`src/views/pokayoke/`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — API 封装** (`src/apis/pokayokeRule.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — Pinia 状态管理** (`src/stores/pokayokeRule.ts`)
  - 编译失败：vue-i18n 依赖缺失
- [ ] **前端 — 路由注册 + 国际化**
  - 编译失败：vue-i18n 依赖缺失

#### 流程引擎（按决策结果）
- [ ] 根据设计决策，实现或重命名流程引擎功能
- [ ] **前端 — 流程模板列表/设计器页** (`src/views/processTemplate/`)
- [ ] **前端 — 流程实例管理页** (`src/views/processInstance/`)
- [ ] **前端 — API 封装 + 状态管理 + 路由 + 国际化**

---

### 四、报表与看板 (SPEC 3.6)

**审查发现问题**
- **`reporting-service` 不可运行** — 3 个 Repository 接口无实现类（Spring 启动崩溃），实体类无 ORM 注解
- **`InventoryReportQueryService`/`FinancialReportQueryService` 100% 硬编码 mock 数据**
- **`SalesReportQueryService` 从自身报表循环聚合**
- **无 SQL 迁移文件，无测试**
- **前端仅首页有 mock setTimeout demo，无真实报表/看板模块**

> **判定原因：** 此前被标记为"审查通过 ✅"，但实际检查发现：(1) `reporting-service` 的 3 个 Repository 接口（`SalesReportRepository`/`InventoryReportRepository`/`FinancialReportRepository`）在整个代码库中没有任何实现类，Spring 启动即报 `NoSuchBeanDefinitionException`；(2) 3 个实体类全是普通 POJO，无 `@Entity`/`@Table` 等 ORM 注解，无法持久化；(3) `InventoryReportQueryService` 和 `FinancialReportQueryService` 全部使用 `BigDecimal.valueOf(5000000)` 等硬编码 mock 值，不是真实数据聚合；(4) `SalesReportQueryService.generateDailyReport()` 通过 `repository.findByDateRange()` 从自身表读取已有报表来"聚合"——循环依赖，无报表时永远为空；(5) 无任何 Flyway 迁移文件、无单元测试。这属于**伪代码/不可运行**，按 CHECK_RULE.md 标准判定为**未完成**。

### 子任务

#### 后端 — reporting-service 重写
- [ ] **实体类添加 ORM 映射**
  - `SalesReport.java`, `InventoryReport.java`, `FinancialReport.java` 添加 `@Entity`/`@Table`/`@Id`/`@Column` 注解
- [ ] **创建 MyBatis-Plus DO + Mapper**
  - 为 3 个报表实体创建 `XxxReportDO.java` + `XxxReportMapper.java`
- [ ] **创建 Repository 接口实现**
  - `SalesReportRepositoryImpl`, `InventoryReportRepositoryImpl`, `FinancialReportRepositoryImpl`
- [ ] **创建 DB 迁移文件**
  - Flyway `V1__reporting_init.sql`，建 `rp_sales_report`/`rp_inventory_report`/`rp_financial_report` 表
- [ ] **重写数据聚合逻辑**
  - `SalesReportQueryService`: 从真实订单数据源（order-service 或订单表）聚合
  - `InventoryReportQueryService`: 从真实库存数据源聚合（非硬编码）
  - `FinancialReportQueryService`: 从真实财务数据源聚合（非硬编码）
- [ ] **后端集成测试**
  - 为 3 个 QueryService 添加 `@SpringBootTest` 集成测试

#### 后端 — mes-service 报表/看板完善
- [ ] **确认 DashboardDesignerService + ReportDesignerService 功能完整**
  - 现有 CRUD 模板管理已就绪 ✅ 验证
- [ ] **添加看板数据聚合接口**
  - 生产看板：产出/良率/工时
  - 质量看板：不良率/缺陷分布
  - 设备看板：OEE/故障率/运行状态
  - 从真实 MES 数据聚合（非硬编码）

#### 前端 — 报表模块
- [ ] **API 封装** (`src/apis/report.ts`)
- [ ] **报表列表页** — 展示已生成报表，支持按类型/日期筛选
- [ ] **报表配置页** — 配置报表模板、数据源、调度频率
- [ ] **报表查看页** — 渲染报表数据（表格 + 图表）
- [ ] **路由注册 + 状态管理 + 国际化**

#### 前端 — 看板模块
- [ ] **API 封装** (`src/apis/dashboard.ts`)
- [ ] **看板设计器** (`src/views/dashboard/designer.vue`)
  - 拖拽布局组件（KPI 卡片、折线图、柱状图、饼图、仪表盘、数据表格）
  - 保存/加载看板模板
- [ ] **生产看板** (`src/views/dashboard/production.vue`)
  - 实时产量、良率、工时、异常告警
- [ ] **质量看板** (`src/views/dashboard/quality.vue`)
  - 不良率趋势、缺陷柏拉图、SPC 控制图
- [ ] **设备看板** (`src/views/dashboard/equipmentDashboard.vue`)
  - OEE 仪表盘、设备状态分布、故障排行
- [ ] **看板展示页** — 基于模板配置动态渲染
- [ ] **路由注册 + 状态管理 + 国际化**

---

## 五、SPEC 方案差距分析（新增需求）

以下为对照 `TODO_MES_SPEC.md` 发现的、当前项目完全缺失或仅为半成品（领域实体无持久化/API）的模块。

> **判定说明：** 这些需求在 SPEC 中明确要求，但实际检查发现当前项目要么完全不存在，要么仅有领域实体却缺少持久化层和 REST API（半成品）。此前未被纳入评审范围，现补充为待办任务。

### 5.1 质量管理 — 4 个半成品补全

领域实体已实现，但 **缺少持久化层和 REST API**：

- [ ] **缺陷代码持久化+API**（`DefectCode.java` 已实现，缺 DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_qc_defect_code`）
- [ ] **检验触发规则持久化+API**（`QcTriggerRule.java` 已实现，缺 DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_qc_trigger_rule`）
- [ ] **不合格处置流程持久化+API**（`NonConformanceDisposition.java` 已实现，缺 DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_qc_non_conformance`）
- [ ] **SPC 控制图持久化+API**（`SpcControlChart.java` 含 7 种控制图和报警规则引擎已实现，缺 DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_qc_spc_control_chart`）

### 5.2 生产执行与报工

- [ ] **创建 WorkReport（报工记录）实体**
  - 包含工时、合格/不合格数量、设备参数值、操作人员、报工时间
  - 绑定到 ProductionTask + ProcessStep
  - DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_work_report`
- [ ] **工时字段启用** — `ProductionTaskDO.actualDurationMinutes` 字段已存在但实体和业务逻辑未使用
- [ ] **设备参数与报工集成** — `ProcessParameter.validateValue()` 在任务完成时未被调用
- [ ] **防错规则集成到任务执行** — `PokaYokeService` 的 `checkMaterial()`/`checkSequence()`/`checkParameter()` 在 `ProductionTaskCommandService.completeTask()` 中未被调用

### 5.3 工位管理

- [ ] **创建 Workstation（工位）实体**
  - 工位编码、名称、所属车间、类型
  - 工位-设备绑定、工位-工具绑定（Tool 实体）、工位-人员绑定
  - DO/Mapper/RepositoryImpl/Controller/DB 表 `mes_workstation`
- [ ] **重构现有 workstationId 字段** — 将 Equipment、ProcessStep 等实体中的 `workstationId`（String）改为关联 Workstation 实体

### 5.4 工艺参数模板

- [ ] **创建 ParameterGroupTemplate（参数组模板）实体**
  - 模板编码、名称、关联产品类型、参数列表
  - DO/Mapper/RepositoryImpl/Controller/DB 表
- [ ] **产品类型到参数组绑定** — 按产品类型自动关联参数组模板

### 5.5 工艺版本完善

- [ ] **ProcessRoute 添加 effectiveDate/expiryDate**
  - 工艺路线的多版本生效日期管理
  - 按日期自动选择有效版本

### 5.6 物料与物流

- [ ] **容器/载具管理** — 完全缺失
  - 容器类型定义、编码规则、状态管理
  - DB 表 + 实体 + API
- [ ] **退料补料流程** — 完全缺失（仅有 `MaterialRequirementItem.cancelIssue()` 基础方法）
  - 退料单、补料单、审批流程
  - DB 表 + 实体 + API

### 5.7 界面配置能力

- [ ] **动态页面布局引擎** — 完全缺失
  - 字段分组、显示顺序、显隐控制、条件显隐
  - 表单设计器后端支持
- [ ] **按钮权限/显隐配置** — 完全缺失（现有仅为方法级 `@RequirePermission` 注解）
  - 按钮按角色/用户组显隐配置
  - 按钮条件可用配置
  - 自定义按钮和动作绑定

### 5.8 系统集成配置

- [ ] **ERP 集成适配器** — 完全缺失
  - 工单同步、BOM 同步、入库回传接口配置
- [ ] **设备集成适配器（OPC UA/Modbus）** — 完全缺失
  - 协议适配、设备驱动配置、地址映射
- [ ] **PLM 集成适配器** — 完全缺失
  - BOM/工艺路线同步
- [ ] **WMS 集成适配器** — 完全缺失
  - 出入库信息交互
- [ ] **企业微信/钉钉消息推送** — 完全缺失
  - 消息模板配置、推送渠道配置

### 5.9 产品类型实体

- [ ] **创建 ProductType（产品类型）实体**
  - 产品分类定义，关联默认工艺路线、质检方案
  - 属性继承（子类型继承父类型配置）
  - 当前 `applicableProductTypes` 仅在 `QcInspectionPlan` 中以字符串字段存在

---

---

## 六、界面国际化补全

以下文件在 `@TODO.md` 需求实现过程中创建或修改，但未使用 `@/locales` 提供的 `t()` 国际化函数，导致大量硬编码中文文本。需逐一补全。

> **说明：** MES 模块（processRoute、equipment、pokayoke）的翻译键已在 `src/locales/` 中定义，仅需修改视图文件；其他模块需同时补充 locale 翻译键和视图文件。

### 6.1 MES 制造执行（翻译键已存在，仅改视图）

- [x] **工艺路线列表页** (`src/views/mes/processRoute/index.vue`) — 41 处中文 ✅
- [ ] **工艺路线表单页** (`src/views/mes/processRoute/form.vue`) — 39 处中文
- [ ] **工艺路线详情页** (`src/views/mes/processRoute/detail.vue`) — 54 处中文
- [ ] **设备列表页** (`src/views/mes/equipment/index.vue`) — 57 处中文
- [ ] **设备表单页** (`src/views/mes/equipment/form.vue`) — 31 处中文
- [ ] **设备详情页** (`src/views/mes/equipment/detail.vue`) — 68 处中文

### 6.2 Poka-Yoke 防错规则（翻译键已存在，仅改视图）

- [ ] **防错规则列表页** (`src/views/pokayoke/index.vue`) — 45 处中文
- [ ] **防错规则表单页** (`src/views/pokayoke/form.vue`) — 36 处中文
- [ ] **防错规则详情页** (`src/views/pokayoke/detail.vue`) — 25 处中文

### 6.3 OMS 订单管理（需补充翻译键 + 改视图）

- [ ] **订单列表页** (`src/views/oms/order/index.vue`) — 108 处中文
- [ ] **订单详情页** (`src/views/oms/order/orderDetail.vue`) — 217 处中文
- [ ] **订单设置页** (`src/views/oms/order/setting.vue`) — 36 处中文
- [ ] **发货列表页** (`src/views/oms/order/deliverOrderList.vue`) — 33 处中文
- [ ] **物流对话框** (`src/views/oms/order/components/logisticsDialog.vue`) — 20 处中文
- [ ] **退货申请列表页** (`src/views/oms/apply/index.vue`) — 57 处中文
- [ ] **退货申请详情页** (`src/views/oms/apply/applyDetail.vue`) — 70 处中文
- [ ] **退货原因管理页** (`src/views/oms/apply/reason.vue`) — 48 处中文

### 6.4 PMS 产品管理（需补充翻译键 + 改视图）

- [ ] **品牌列表页** (`src/views/pms/brand/index.vue`) — 63 处中文
- [ ] **品牌详情组件** (`src/views/pms/brand/components/BrandDetail.vue`) — 42 处中文
- [ ] **商品详情组件** (`src/views/pms/product/components/ProductDetail.vue`) — 27 处中文
- [ ] **商品信息详情组件** (`src/views/pms/product/components/ProductInfoDetail.vue`) — 48 处中文
- [ ] **商品属性详情组件** (`src/views/pms/product/components/ProductAttrDetail.vue`) — 85 处中文
- [ ] **商品关联详情组件** (`src/views/pms/product/components/ProductRelationDetail.vue`) — 25 处中文
- [ ] **商品促销详情组件** (`src/views/pms/product/components/ProductSaleDetail.vue`) — 58 处中文
- [ ] **属性列表页** (`src/views/pms/productAttr/index.vue`) — 50 处中文
- [ ] **属性列表组件** (`src/views/pms/productAttr/productAttrList.vue`) — 48 处中文
- [ ] **属性详情组件** (`src/views/pms/productAttr/components/ProductAttrDetail.vue`) — 55 处中文
- [ ] **分类列表页** (`src/views/pms/productCate/index.vue`) — 46 处中文
- [ ] **分类详情组件** (`src/views/pms/productCate/components/ProductCateDetail.vue`) — 57 处中文

### 6.5 SMS 营销管理（需补充翻译键 + 改视图）

- [ ] **广告列表页** (`src/views/sms/advertise/index.vue`) — 69 处中文
- [ ] **广告详情组件** (`src/views/sms/advertise/components/HomeAdvertiseDetail.vue`) — 46 处中文
- [ ] **品牌推荐页** (`src/views/sms/brand/index.vue`) — 105 处中文
- [ ] **优惠券列表页** (`src/views/sms/coupon/index.vue`) — 60 处中文
- [ ] **优惠券详情组件** (`src/views/sms/coupon/components/CouponDetail.vue`) — 92 处中文
- [ ] **优惠券使用历史页** (`src/views/sms/coupon/history.vue`) — 65 处中文
- [ ] **秒杀活动列表页** (`src/views/sms/flash/index.vue`) — 80 处中文
- [ ] **秒杀商品关联页** (`src/views/sms/flash/productRelationList.vue`) — 80 处中文
- [ ] **秒杀时间段列表页** (`src/views/sms/flash/sessionList.vue`) — 57 处中文
- [ ] **秒杀场次选择页** (`src/views/sms/flash/selectSessionList.vue`) — 15 处中文
- [ ] **人气推荐页** (`src/views/sms/hot/index.vue`) — 103 处中文
- [ ] **新品推荐页** (`src/views/sms/new/index.vue`) — 103 处中文
- [ ] **专题推荐页** (`src/views/sms/subject/index.vue`) — 101 处中文

### 6.6 UMS 用户管理（需补充翻译键 + 改视图）

- [ ] **管理员列表页** (`src/views/ums/admin/index.vue`) — 100 处中文
- [ ] **菜单详情组件** (`src/views/ums/menu/components/MenuDetail.vue`) — 42 处中文
- [ ] **资源分类列表页** (`src/views/ums/resource/categoryList.vue`) — 44 处中文
- [ ] **资源列表页** (`src/views/ums/resource/index.vue`) — 70 处中文
- [ ] **分配菜单页** (`src/views/ums/role/allocMenu.vue`) — 19 处中文
- [ ] **分配资源页** (`src/views/ums/role/allocResource.vue`) — 24 处中文

### 6.7 其他模块（需补充翻译键 + 改视图）

- [ ] **客服人员页** (`src/views/cs/agents.vue`) — 21 处中文
- [ ] **客服仪表盘** (`src/views/cs/dashboard.vue`) — 28 处中文
- [ ] **快捷回复管理页** (`src/views/cs/quick-reply.vue`) — 20 处中文
- [ ] **首页仪表盘** (`src/views/home/index.vue`) — 83 处中文
- [ ] **404 页面** (`src/views/normal/404/index.vue`) — 8 处中文
- [ ] **登录页** (`src/views/normal/login/index.vue`) — 17 处中文

### 6.8 布局组件（需补充翻译键 + 改视图）

- [ ] **AppMain 组件** (`src/views/layout/components/AppMain.vue`)
- [ ] **Navbar 组件** (`src/views/layout/components/Navbar.vue`) — 含 "首页"、"退出"等
- [ ] **Sidebar 组件** (`src/views/layout/components/Sidebar/index.vue`)
- [ ] **SidebarItem 组件** (`src/views/layout/components/Sidebar/SidebarItem.vue`)
- [ ] **Layout 主布局** (`src/views/layout/Layout.vue`)

### 6.9 部分国际化文件补全（已有 i18n 导入，仍有残留中文）

- [ ] **角色列表页** (`src/views/ums/role/index.vue`) — 49 处残留中文
- [ ] **菜单列表页** (`src/views/ums/menu/index.vue`) — 24 处残留中文
- [ ] 检查 @server/ 里是否有使用中文作为文本，需要全部改成英文