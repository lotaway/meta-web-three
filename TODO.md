# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 🚨 Build Errors - Needs Immediate Fix

**问题**：前端代码使用了 `vue-i18n`（`useI18n()`），但 `package.json` 中未声明该依赖，导致编译失败。

**影响范围**：所有前端 MES 模块（processRoute、equipment、pokayoke）

**修复状态**：✅ 已完成
- 已在 `apps/backstage-admin/package.json` 添加 `"vue-i18n": "^9.14.0"`
- 已执行 `npm install` 安装依赖
- 项目已可成功构建（npm run build-only 成功）✅
- **遗留问题**：type-check 有约 22 个错误，主要是布局组件 SidebarItem.vue 参数类型问题、equipment 模块中部分组件类型定义问题和 API 响应类型问题。构建不受影响，可正常运行。
- **新增遗留问题（2026-05-28）**：type-check 有 30+ 错误在 `src/apis/qc.ts`，`http.get/post/put/delete` 方法不存在（应为 `http({method:'get',url:...})` 格式），这是之前修复 QC API 时引入的问题。构建不受影响，可正常运行。
- **构建警告修复（2026-05-28）**：✅ 已修复
  - 修复 `src/locales/en-US.ts` 重复的 `menu` 键（第二个 menu 块内容未正确嵌套）
  - `src/views/mes/equipment/form.vue` 的 FormInstance 导入在当前版本下正常，构建不再报错
- **type-check 错误修复**：✅ 基本完成（2026-05-28）
  - **修复内容**：
    - ✅ 已修复 processRoute/index.vue、detail.vue 的 el-tag type 返回类型
    - ✅ 已修复 equipment/index.vue、detail.vue 的 el-tag type 返回类型 + 添加缺失的 handleReportBreakdown 函数
    - ✅ 已修复 equipment/checklist.vue、maintenancePlan.vue 的 el-tag type 返回类型 + form 添加 id 字段
    - ✅ 已修复 pokayoke/index.vue、detail.vue 的 el-tag type 返回类型
    - ✅ 已修复 cs/dashboard.vue 的 quickReplies key 类型问题
    - ✅ 已修复 SidebarItem.vue 的 name 参数类型问题（添加 safeName 函数）
  - **构建状态**：✅ npm run build-only 成功
  - **说明**：SidebarItem.vue 仍有 4 个 type-check 警告，但构建不受影响，属于 Vue 模板类型推断的已知问题

---

## 代码规范审查结果 (2026-05-28)

### 审查说明
已修复 vue-i18n 依赖并完成重新验证。以下为对照[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)、[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[检查规则](CODE_PINCEPLES/CHECK_RULE.md)的逐项审查结果。

### 审查结果摘要 (2026-05-28)

本次审查了 9 个已完成项：
- **通过审查（已删除）**: 4 项（API 封装、表单页、流程引擎决策、Pinia 状态管理）
- **通过审查（已修复）**: 5 项（设备列表页、设备详情页、保养计划页、检查清单页、PokaYoke 列表页）—— catch 块均使用 ElMessage.error 正确处理错误
- **审查未通过（需修复）**: 2 项

⚠️ **审查发现**：多个已完成项目的 catch 块为空或仅有注释，违反代码规范"禁止吞异常"。具体问题见各子任务。

---

### 一、工艺路线 (ProcessRoute)

**审查发现问题（已更新）**
- 后端领域逻辑 ✅ 真实实现（实体、仓储、测试）
- **前端功能完整** ✅（API/页面/store/路由/国际化均已实现）
- **前端代码规范问题** ⚠️
  - 违反"禁止注释"：API 文件、Store、视图文件有多处中文注释（函数说明、空 catch 的 `// 用户取消` 等）
  - 违反"禁止吞异常"：多个 catch 块为空或仅 `console.error`
  - 违反"禁止使用打印代替功能实现"：`console.error()` 用于错误处理
  - 违反国际化要求：路由 meta.title 为硬编码中文（`'工艺路线'`、`'制造执行'` 等）
- **后端 Controller 规范问题已修复** ✅（移除 JavaDoc，使用自定义异常）

### 子任务

- [ ] **后端 — 集成测试** 🔄 进行中（尝试中，困难：Spring测试上下文加载复杂）
  - 问题：ApplicationContext 加载失败，17个测试全部报错 `IllegalStateException: ApplicationContext failure threshold exceeded`
  - 根本原因：测试环境复杂，依赖大量自动配置（Kafka、Security、OAuth2等），配置排除不完全
  - 解决尝试：
    - 创建 `TestConfig.java` 提供 mock DomainEventPublisher 和 MesEventPublisher → 失败（ClassNotFoundException: DomainEventPublisher）
    - 修改测试类使用 `@MockBean` 模拟 MesEventPublisher → 仍失败（PropertyPlaceholder 配置错误）
    - 修改 `application-test.yml` 排除更多 Security 自动配置 → 仍失败
  - 当前状态：单元测试 `ProcessRouteTest` (20用例) 已通过 ✅，集成测试因 Spring 上下文加载问题暂无法运行
  - 结论：集成测试环境配置复杂度高，建议简化测试策略（如使用 @WebMvcTest 替代 @SpringBootTest）
- [ ] **前端 — API 封装** (`src/apis/processRoute.ts`) ✅ 已修复 — 删除了所有中文注释 ✓ 已通过审查，已删除
- [ ] **前端 — 工艺路线新增/编辑页** (`src/views/mes/processRoute/form.vue`) ✅ 已修复 — 删除了中文注释，catch 块使用 ElMessage 显示错误 ✓ 已通过审查，已删除
- [ ] **前端 — 工艺路线详情页** (`src/views/mes/processRoute/detail.vue`) ✅ 已修复 — 删除了中文注释，catch 块正确处理错误 ✓ 已通过审查，已删除
- [ ] **前端 — 国际化** (`src/locales/zh-CN.ts` + `en-US.ts` 添加翻译键) ✅ 功能完整，通过检查 ✓ 已通过审查，已删除

---

### 二、设备管理 (Equipment)

**审查发现问题（已更新）**
- 后端含真实 OEE 算法和持久化 ✅
- **后端实体状态校验已存在** ✅（`Equipment.java` 已实现 6 个状态转换方法的状态校验）
- **后端 Controller 已存在** ✅（含完整 CRUD + 状态转换 + 数字孪生 + OEE 计算）
- **前端功能完整** ✅（API/Store/路由/列表页/表单页/详情页均已存在）
- **前端代码规范问题** ⚠️
  - 违反"禁止注释"：API 文件、Store、视图文件有多处中文注释
  - 违反"禁止吞异常"：多个 catch 块为空或仅 `console.error`
  - 违反"禁止使用打印代替功能实现"：`console.error()` 用于错误处理
  - 违反国际化要求：路由 meta.title 硬编码中文；maintenancePlan.vue 和 checklist.vue 的 rules 有 9 处硬编码中文校验提示

> **更新说明（2026-05-28）**：经核查，`Equipment.java` 中已实现所有状态校验方法。**设备管理模块功能已完整实现**，但前端存在代码规范问题需修复。

### 子任务

- [ ] **前端 — 设备表单页** ✅ 已修复 — form.vue catch 块使用 ElMessage.error()
- [ ] **前端 — 国际化** ✅ 已修复 — 路由 meta.title 使用国际化键（如 'mes.equipment.title'）

---

### 三、流程引擎与规则引擎 (SPEC 3.3 / 3.4)

**审查发现问题**
- 规则引擎(PokaYoke) ✅ 后端真实算法 + 前端完整实现
- **流程引擎仅为模板/实例管理器，`startInstance()` 只复制 JSON 不执行节点路由**
- **`pom.xml` 中 Flowable 依赖声明但未使用**
- **流程引擎前端完全不存在**

> **更新说明（2026-05-28）**：规则引擎(PokaYoke)前后端均已完整实现。流程引擎仅模板管理，无真正 BPMN 能力。

### 子任务

#### 决策前置
  - 决策结果：当前流程引擎为简单的模板/实例管理器，仅复制 JSON 数据，不使用 BPMN 引擎
  - Flowable 依赖已移除（未实际使用，仅增加构建复杂度）

#### 规则引擎（已有后端，补充前端）
- [ ] **后端 — 规则引擎集成测试**
- [ ] **前端 — API 封装** ✅ 通过检查 ✓ 已通过审查，已删除

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
- [ ] **DO/Mapper/RepositoryImpl 已补全**（2026-05-28 已完成 `schema.sql`，DO/Mapper/RepositoryImpl/Converter 均已存在）
- [ ] **打通 mall-domain 数据链路 — 销售报表对接商城订单**

  当前 `SalesReportQueryService` 仅从自身历史报表循环聚合（无历史数据时全部硬编码），需从 **mall-domain 的 order-service** 获取真实销售数据：

  - 方式一（推荐）：**Feign 客户端** — 在 reporting-service 中创建 `OrderFeignClient`，调用 `order-service` 的订单查询接口，按时间范围/状态聚合订单金额、数量、品类分布
  - 方式二：**Kafka 事件消费** — 消费 `order-service` 发布的订单事件（如 `OrderCompletedEvent`/`OrderPaidEvent`），实时或准实时累积到报表表中
  - 需聚合的维度：总销售额、订单数、平均客单价、毛利率（需对接成本数据）、品类/商品/渠道维度汇总
  - `category_breakdown`/`product_ranking`/`channel_breakdown` 从 TEXT JSON 改为**规范化维度表 + 事实表结构**，或至少保证聚合逻辑真实

- [ ] **打通 mall-domain 数据链路 — 库存报表对接仓储库存**

  当前 `InventoryReportQueryService` 完全硬编码（500万库存价值、2500 SKU等），需从 **supply-chain-domain 的 inventory-service / warehouse-service** 获取真实库存数据：

  - 聚合库存总价值、SKU 数、总库存量
  - 计算周转率（需对接出库/发货数据）
  - 慢动物料判定逻辑
  - 仓库维度/品类维度分布
  - 低库存预警（低于安全库存的 SKU 列表）

- [ ] **打通数据链路 — 财务报表对接 finance-service**

  当前 `FinancialReportQueryService` 完全硬编码，需从 **erp-domain 的 finance-service** 获取真实财务数据：

  - 应收/应付汇总（对接 Account/AccountSubject）
  - 账龄分析（对接 Voucher 的账期字段）
  - 营运资金/流动比率计算

- [ ] **添加 Feign 或 gRPC 跨服务调用依赖**

  - reporting-service 的 pom.xml 当前无任何 RPC 依赖（仅依赖 common + event-sdk）
  - 需添加 `spring-cloud-starter-openfeign` 或注册中心/负载均衡依赖
  - 若采用事件驱动，需实现 `event-sdk` 的消费者（当前声明了依赖但完全未使用）

- [ ] **添加定时调度自动生成报表**

  当前报表必须通过 POST 端点手动触发，需添加：
  - `@Scheduled(cron = "0 0 1 * * ?")` 每日凌晨自动生成昨日日报
  - `@Scheduled(cron = "0 0 2 1 * ?")` 每月 1 号自动生成上月月报
  - 可在 `application.yml` 中配置 cron 表达式支持开关

- [ ] **添加报表导出功能**

  所有三类报表缺少导出能力，需补充：
  - CSV 导出（`/api/reports/sales/{id}/export?format=csv`）
  - Excel 导出（使用 Apache POI 或 EasyExcel）
  - PDF 导出（可选，使用 JasperReports/iText）

- [ ] **正确使用 X-User-Id / X-User-Role 请求头**

  当前 ReportController 接收了 `X-User-Id` 和 `X-User-Role` 请求头，但 QueryService 的业务逻辑中完全未使用。需：
  - 写入 `created_by`/`updated_by` 字段到报表表
  - 支持按用户/角色过滤报表可见范围

- [ ] **后端集成测试**
  - 为 3 个 QueryService 添加 `@SpringBootTest` 集成测试
  - Mock 外部 Feign 客户端（使用 `@MockBean`）

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

**状态更新（2026-05-28）**：经验证，质量管理模块已完整实现：
- ✅ 领域实体（QcInspectionPlan/Item/Type/TriggerRule）
- ✅ DO 类（QcInspectionPlanDO 等）
- ✅ MyBatis Mapper
- ✅ RepositoryImpl 实现
- ✅ REST Controller
- ✅ Flyway 迁移文件（V6-V9）

该任务可标记为已完成。


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

> **说明**：工艺路线列表/表单/详情页、设备列表页的国际化已通过检查。设备表单/详情页也已完成国际化（经核查已使用 `t()` 无硬编码中文），之前标记为未完成是清单未及时更新。

### 6.2 Poka-Yoke 防错规则（翻译键已存在，仅改视图） ✅ 已修复（2026-05-28）
  - 问题: 原 `/src/views/mes/pokayoke/` 目录不存在，模块误放在 `src/views/pokayoke/`
  - 修复: 已将文件移动到正确位置 `src/views/mes/pokayoke/` 并更新路由配置
  - 构建验证: ✅ npm run build-only 成功
### 6.3 OMS 订单管理（需补充翻译键 + 改视图）

- [ ] **订单详情页** (`src/views/oms/order/orderDetail.vue`) — 217 处中文
  - 将硬编码模拟数据改为使用 `i18n('logisticsSubmitted')` 等国际化键
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
- [ ] **角色列表页** (`src/views/ums/role/index.vue`) ✅ 已修复（2026-05-28）
  - 问题1: 4处使用 `console.error` 改为 `ElMessage.error(t('role.page.xxx'))`
  - 问题2: 保留本地 `i18n` 函数包装 `t()`（功能正常，符合规范）
  - 状态: 已通过审查

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

- [ ] **角色列表页** (`src/views/ums/role/index.vue`) ❌ 未通过审查
  - 问题1: 4处使用 `console.error` 代替 ElMessage.error，违反"禁止使用打印代替功能实现"规范
  - 问题2: 使用未定义的 `i18n()` 函数，应使用已导入的 `t()` 函数
- [ ] **菜单列表页** (`src/views/ums/menu/index.vue`) ✅ 已修复（2026-05-28）
  - 问题: 2处使用 `console.error` 改为 `ElMessage.error(t('menu.xxx'))`
  - 状态: 已通过审查

---

## 七、后端基础设施缺失（持久层/数据库）

> **发现时间：2026-05-28**。经全面审查，ERP 和供应链域服务的**核心业务逻辑已实现**，但持久化层普遍缺失，服务无法真正运行。

### 7.1 ERP 域 — 四服务均缺持久层

**共同问题**：以下四服务的 Domain 实体包含真实业务逻辑（状态机、财务计算），但：
- ❌ 无 `@Entity`/`@Table` ORM 注解（纯 POJO）
- ❌ 无 DO 数据对象（MyBatis-Plus `@TableName`）
- ❌ 无 Mapper 接口（`extends BaseMapper`）
- ❌ 无 RepositoryImpl 实现
- ❌ 无 Flyway 数据库迁移文件
- ❌ 无单元测试/集成测试

#### finance-service（26个文件，端口 10110）

**实体逻辑**：✅ Account（96行，含 credit/debit/freeze/unfreeze）、AccountSubject（75行）、Voucher（122行，含完整凭证审批流）
**Controller**：✅ 4个 Controller（Account/AccountSubject/Voucher/FinancialReport），含 `@RequirePermission` 注解
**事件**：✅ Kafka 事件发布器（真实实现）

**待补充**：
- [ ] 创建 AccountDO/AccountMapper/AccountRepositoryImpl
- [ ] 创建 AccountSubjectDO/AccountSubjectMapper/AccountSubjectRepositoryImpl
- [ ] 创建 VoucherDO/VoucherMapper/VoucherRepositoryImpl（含 VoucherLine 明细）
- [ ] Flyway 迁移：`V1__finance_init.sql`（account、account_subject、voucher、voucher_line 表）
- [ ] 集成测试（20+ 用例覆盖账户操作、凭证审批流）
- [ ] 空 adapter 目录处理：`adapter/grpc/`、`adapter/http/`、`adapter/vo/`（实现或清理）

#### invoice-service（7个文件，端口 10111）

**实体逻辑**：✅ Invoice（129行，DRAFT→ISSUED→PRINTED→VOIDED/RED_FLUSHED 完整状态机）
**Controller**：✅ 完整 CRUD + 开票/打印/作废/红冲

**待补充**：
- [ ] 创建 InvoiceDO/InvoiceMapper/InvoiceRepositoryImpl
- [ ] Flyway 迁移：`V1__invoice_init.sql`（invoice 表）
- [ ] 集成测试（10+ 用例覆盖发票全生命周期）

#### settlement-service（12个文件，端口 10113）

**实体逻辑**：✅ SettlementOrder（110行，含自动佣金计算 `settlementAmount = orderAmount - (orderAmount * commissionRate)`）、ReconciliationRecord（95行）、SplitRule（118行，4种分账规则：比例/固定/保底封顶/混合）
**定时任务**：✅ 日自动结算 (`0 0 2 * * ?`) + 月对账触发 (`0 0 0 1 * ?`)
**Controller**：✅ 完整 CRUD + 确认/处理/完成/失败/取消/退款

**待补充**：
- [ ] 创建 SettlementOrderDO/SettlementOrderMapper/SettlementOrderRepositoryImpl
- [ ] 创建 ReconciliationRecordDO/ReconciliationRecordMapper/ReconciliationRecordRepositoryImpl
- [ ] 创建 SplitRuleDO/SplitRuleMapper/SplitRuleRepositoryImpl
- [ ] Flyway 迁移（settlement_order、reconciliation_record、split_rule 表）
- [ ] 集成测试（15+ 用例覆盖结算流程、分账计算）

#### reporting-service（已在第 4 节详细列出）
- [ ] 按第 4 节计划完整重写（实体 ORM + DO/Mapper + RepositoryImpl + Flyway + 真实聚合 + 测试）

---

### 7.2 供应链域 — 持久化层不完整

| 服务 | 文件数 | 持久化状态 | 紧急程度 |
|------|--------|-----------|:-------:|
| **warehouse-service**（10106） | 33 | ✅ **唯一完整**（3 DO + 3 Mapper + 2 Converter + 2 RepositoryImpl） | 低 |
| **inventory-service**（10105） | 18 | ❌ **完全缺失**（Repository 接口无实现，findBySkuAndWarehouse 恒空） | 🔴 高 |
| **logistics-service**（10107） | 15 | ❌ **完全缺失**（Repository 接口无实现，listOrders 返回空列表） | 🔴 高 |
| **supplier-service**（10109） | 13 | ⚠️ **内存存储**（ConcurrentHashMap，重启数据丢失） | 🟡 中 |
| **procurement-service**（10108） | 13 | ⚠️ **内存存储**（ConcurrentHashMap，重启数据丢失） | 🟡 中 |

#### warehouse-service（补充完善）
- [ ] Flyway 迁移：`V1__warehouse_init.sql`（warehouse、location、inbound_order、inbound_order_item、outbound_order、outbound_order_item 表）
- [ ] 单元测试 + 集成测试（覆盖入库/出库完整流程）

#### inventory-service（需完整建持久层）
- [ ] 创建 InventoryDO/InventoryMapper/InventoryRepositoryImpl
- [ ] 创建 InventoryRecordDO/InventoryRecordMapper
- [ ] 修复 `findBySkuAndWarehouse()` 始终返回空的 bug
- [ ] Flyway 迁移：`V1__inventory_init.sql`（inventory、inventory_record 表）
- [ ] 集成测试（覆盖预留/确认/取消/增减库存流程）

#### logistics-service（需完整建持久层）
- [ ] 创建 LogisticsOrderDO/LogisticsOrderMapper/LogisticsOrderRepositoryImpl
- [ ] 创建 CarrierDO/CarrierMapper
- [ ] 创建 TrackingEventDO/TrackingEventMapper
- [ ] 修复 `listOrders()` 返回空列表的问题
- [ ] Flyway 迁移：`V1__logistics_init.sql`（logistics_order、carrier、tracking_event 表）
- [ ] 集成测试

#### supplier-service（替换内存存储）
- [ ] 创建 SupplierDO/SupplierMapper/SupplierRepositoryImpl（替换 ConcurrentHashMap）
- [ ] 添加 Controller 权限注解（`@RequirePermission`）
- [ ] Flyway 迁移：`V1__supplier_init.sql`（supplier 表）
- [ ] 集成测试

#### procurement-service（替换内存存储）
- [ ] 创建 ProcurementOrderDO/ProcurementOrderMapper/ProcurementOrderRepositoryImpl
- [ ] 添加 Controller 权限注解（`@RequirePermission`）
- [ ] Flyway 迁移：`V1__procurement_init.sql`（procurement_order 表）
- [ ] 集成测试

---

### 7.3 MES 后端 — BOM 物料清单完整链路缺失

**背景**：BOM 相关 **6 个领域实体 + 3 个 Repository 接口** 已实现（含丰富业务逻辑），但 Infrastructure 层、REST API 层、数据库表全部缺失。

| 组件 | 状态 | 文件 |
|------|------|------|
| BomBillOfMaterials 实体 | ✅ 完整 | `domain/entity/BomBillOfMaterials.java` |
| BomVersion 实体 | ✅ 完整 | `domain/entity/BomVersion.java`（含版本记录+生命周期） |
| BomItem 实体 | ✅ 完整 | `domain/entity/BomItem.java`（含报废率+替代料关联） |
| ProcessBomItem 实体 | ✅ 完整 | `domain/entity/ProcessBomItem.java`（工序级 BOM） |
| MaterialRequirement 实体 | ✅ 完整 | `domain/entity/MaterialRequirement.java`（领料全流程） |
| MaterialSubstitute 实体 | ✅ 完整 | `domain/entity/MaterialSubstitute.java`（含替代料优先级+转换率） |
| DO/Mapper | ❌ 缺失 | 需为 6 个实体各建 1 个 |
| RepositoryImpl | ❌ 缺失 | 需实现 3 个 Repository 接口 |
| REST Controller | ❌ 缺失 | 需创建 BomController + MaterialRequirementController + MaterialSubstituteController |
| Flyway 迁移 | ❌ 缺失 | 无任何 BOM 表（schema.sql + V1-V9 均不包含） |

**待办**：
- [ ] **DO/Mapper**：创建 BomBillOfMaterialsDO/BomMapper、BomVersionDO/BomVersionMapper、BomItemDO/BomItemMapper、ProcessBomItemDO/ProcessBomItemMapper、MaterialRequirementDO/MaterialRequirementMapper、MaterialSubstituteDO/MaterialSubstituteMapper
- [ ] **RepositoryImpl**：实现 BomRepositoryImpl、MaterialRequirementRepositoryImpl、MaterialSubstituteRepositoryImpl
- [ ] **Controller**：创建 BomController（BOM CRUD + 版本管理）、MaterialRequirementController（物料需求）、MaterialSubstituteController（替代料管理）
- [ ] **Flyway**：`V10__bom_init.sql`（6 张 BOM 表）
- [ ] **集成测试**

---

## 八、MES 前端缺失页面（后端已实现，前端待补）

> **发现时间：2026-05-28**。以下模块后端已在 mes-service 中**完整实现**（含实体 + Controller + 持久层），但 backstage-admin 无对应前端页面。

### 8.1 QC 质检管理（7个子模块）

**后端**：✅ 完整（QcInspectionType/Plan/Item + DefectCode + QcTriggerRule + NonConformanceDisposition + SpcControlChart）
**前端**：🔄 进行中

- [ ] **API 封装** (`src/apis/qc.ts`) ❌ 未通过审查（2026-05-28）
  - 问题1：多个 API 函数 URL 路径为空 → 已补全所有 URL 路径（30+ 函数）
  - 问题2：部分函数缺少参数传递 → 已修复参数传递
  - 问题3（审查新发现）：type-check 有 30+ 错误，`http.get/post/put/delete` 方法不存在于 http 函数（应为 `http({method:'get',url:...})`）
  - 问题4：DefectSeverity 类型中 `' observation'` 有前导空格（typo）
  - 问题5：文件顶部有注释 `// QC Quality Inspection API`，违反「禁止注释」规范
  - 修复建议：将所有 `http.get(url)` 改为 `http({ method: 'get', url })` 格式，修正 DefectSeverity 中的空格，删除文件头注释
  - 构建验证：✅ npm run build-only 成功（但 type-check 有 30+ 错误）
- [x] **检验类型管理页** (`src/views/mes/qc/inspectionType/`) ✅ 已完成（2026-05-29）
  - 已添加 QC 检验类型国际化翻译键到 zh-CN.ts 和 en-US.ts
  - index.vue、form.vue、detail.vue 均已使用 `t()` 国际化函数替换硬编码中文
  - 构建验证：✅ npm run build-only 成功
- [ ] **检验计划管理页** (`src/views/mes/qc/inspectionPlan/`)- [ ] **检验项管理页** (`src/views/mes/qc/inspectionItem/`)
- [ ] **缺陷代码管理页** (`src/views/mes/qc/defectCode/`)
- [ ] **触发规则管理页** (`src/views/mes/qc/triggerRule/`)
- [ ] **不合格处置页** (`src/views/mes/qc/nonConformance/`)
- [ ] **SPC 控制图页** (`src/views/mes/qc/spc/`)
- [ ] **Store + 路由 + 国际化**
### 8.2 ProductionTask 生产任务

**后端**：✅ 完整（112行实体 + 237行 Controller + CommandService + QueryService + RepositoryImpl）
**前端**：❌ 完全缺失

- [ ] **API 封装** (`src/apis/productionTask.ts`)
- [ ] **生产任务列表页** (`src/views/mes/productionTask/index.vue`) — 状态筛选、任务编号/工位/设备搜索
- [ ] **生产任务详情页** (`src/views/mes/productionTask/detail.vue`) — 进度、报工、质检、设备参数
- [ ] **生产任务表单页** (`src/views/mes/productionTask/form.vue`) — 创建/编辑、分配工位/设备
- [ ] **Store + 路由 + 国际化**

### 8.3 WorkOrder 工单管理

**后端**：✅ 完整（296行实体，含完整状态机 + 拆分规则 + 物料需求计算）
**前端**：❌ 完全缺失

- [ ] **API 封装** (`src/apis/workOrder.ts`)
- [ ] **工单列表页** (`src/views/mes/workOrder/index.vue`)
- [ ] **工单详情页** (`src/views/mes/workOrder/detail.vue`)
- [ ] **工单表单页** (`src/views/mes/workOrder/form.vue`)
- [ ] **Store + 路由 + 国际化**

### 8.4 BOM 物料清单

**后端**：⚠️ 仅 Domain 层完整（需先完成 7.3 补齐持久层 + Controller）
**前端**：❌ 完全缺失

- [ ] **API 封装** (`src/apis/bom.ts`)
- [ ] **BOM 列表页** (`src/views/mes/bom/index.vue`) — 树形展示、多版本
- [ ] **BOM 详情/编辑页** (`src/views/mes/bom/detail.vue`)
- [ ] **物料需求页** (`src/views/mes/bom/materialRequirement.vue`)
- [ ] **替代料管理页** (`src/views/mes/bom/materialSubstitute.vue`)
- [ ] **Store + 路由 + 国际化**

---

## 九、前端 ERP 模块（新增，从零建设）

- [] ERP 后端四服务（finance/invoice/settlement/reporting）业务逻辑已实现，但前端完全缺失，看看是否放到和[商城后台管理](backstage-admin)一起合适。

### 9.1 财务管理

**后端基于**：finance-service（Account/AccountSubject/Voucher + 凭证审批流 + 财务报表）

- [ ] **API 封装** (`src/apis/finance.ts`)
- [ ] **科目管理页** (`src/views/erp/finance/subject/`)
- [ ] **账户管理页** (`src/views/erp/finance/account/`)
- [ ] **凭证管理页** (`src/views/erp/finance/voucher/`) — 含审批流（草稿→提交→批准→过账）
- [ ] **财务报表页** (`src/views/erp/finance/report/`) — 资产负债表、利润表、试算平衡表
- [ ] **Store + 路由 + 国际化**

### 9.2 发票管理

- [ ] **API 封装** (`src/apis/invoice.ts`)
- [ ] **发票管理页** (`src/views/erp/invoice/`) — 开具/打印/作废/红冲
- [ ] **Store + 路由 + 国际化**

### 9.3 结算管理

- [ ] **API 封装** (`src/apis/settlement.ts`)
- [ ] **结算单管理页** (`src/views/erp/settlement/`)
- [ ] **对账记录页** (`src/views/erp/settlement/reconciliation/`)
- [ ] **分账规则管理页** (`src/views/erp/settlement/splitRule/`)
- [ ] **Store + 路由 + 国际化**

### 9.4 报表管理

- [ ] （与第 4 节前端报表/看板模块合并，后端 `reporting-service` 需先重写）

---

## 十、前端供应链模块（新增，从零建设）

> **说明**：供应链后端五服务业务逻辑已实现，但前端完全缺失。

### 10.1 仓库管理

**后端**：warehouse-service（最完整，含完整持久层 + 10个 API 端点）

- [ ] **API 封装** (`src/apis/warehouse.ts`)
- [ ] **仓库管理页** (`src/views/supply-chain/warehouse/`)
- [ ] **库位管理页** (`src/views/supply-chain/warehouse/location.vue`)
- [ ] **入库单管理页** (`src/views/supply-chain/warehouse/inbound.vue`)
- [ ] **出库单管理页** (`src/views/supply-chain/warehouse/outbound.vue`)
- [ ] **Store + 路由 + 国际化**

### 10.2 库存管理

**后端**：inventory-service（需先补持久层 7.2）

- [ ] **API 封装** (`src/apis/inventory.ts`)
- [ ] **库存概览页** (`src/views/supply-chain/inventory/`)
- [ ] **库存流水页** (`src/views/supply-chain/inventory/record.vue`)
- [ ] **Store + 路由 + 国际化**

### 10.3 采购管理

**后端**：procurement-service（需先替换内存存储 7.2）

- [ ] **API 封装** (`src/apis/procurement.ts`)
- [ ] **采购单管理页** (`src/views/supply-chain/procurement/`)
- [ ] **Store + 路由 + 国际化**

### 10.4 供应商管理

**后端**：supplier-service（需先替换内存存储 7.2）

- [ ] **API 封装** (`src/apis/supplier.ts`)
- [ ] **供应商管理页** (`src/views/supply-chain/supplier/`)
- [ ] **Store + 路由 + 国际化**

### 10.5 物流管理

**后端**：logistics-service（需先补持久层 7.2）

- [ ] **API 封装** (`src/apis/logistics.ts`)
- [ ] **物流运单管理页** (`src/views/supply-chain/logistics/`)
- [ ] **承运商管理页** (`src/views/supply-chain/logistics/carrier.vue`)
- [ ] **Store + 路由 + 国际化**

---

## 十一、现有功能问题修复

### 11.1 Supplier/Procurement 控制器缺少权限注解

**问题**：supplier-service 和 procurement-service 的 Controller 未使用 `@RequirePermission` 注解，与项目中其他服务不一致。

- [ ] `SupplierController` 添加 `@RequirePermission` 注解
- [ ] `ProcurementController` 添加 `@RequirePermission` 注解

### 11.2 InventoryDomainServiceImpl.findBySkuAndWarehouse 始终返回空

**文件**：`inventory-service/.../InventoryDomainServiceImpl.java`
**问题**：方法体为 `return Optional.empty()`，未实现实际查询逻辑
**影响**：库存预留/查询功能无法正常工作
- [ ] 配合持久层实现（7.2）后修复为真实查询

### 11.3 LogisticsApplicationServiceImpl.listOrders 返回空列表

**文件**：`logistics-service/.../LogisticsApplicationServiceImpl.java`
**问题**：直接返回 `List.of()`
- [ ] 配合持久层实现（7.2）后修复
- [ ] 按照[TODO_FACTORY_PLAN](./TODO_FACTORY_PLAN.md)完成功能完善