# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 🚨 Build Errors - Needs Immediate Fix

**问题**：前端代码使用了 `vue-i18n`（`useI18n()`），但 `package.json` 中未声明该依赖，导致编译失败。

**影响范围**：所有前端 MES 模块（processRoute、equipment、pokayoke）

**修复状态**：✅ **已完成（2026-05-29）**
- ✅ 前端 vue-i18n 依赖已添加，构建成功
- ✅ WorkOrderController 等 12 个 Controller 的 RequirePermission 导入已修复
- ✅ WorkReport 实体添加了 setCreatedAt/setUpdatedAt 方法
- ✅ MesDomainServiceImpl.completeTask 方法参数已修复
- ✅ **reporting-service pom.xml mybatis-plus 版本已修复**（2026-05-29）
  - 添加版本号 `${mybatis-plus.version}`
  - 修复 SalesReport/InventoryReport/FinancialReport 实体缺失的 setter 方法
  - 修复 FinancialReportRepository/InventoryReportRepository 接口缺少 findByDateRange 方法
- ✅ **reporting-service 编译成功**（2026-05-29）
- ✅ **mes-service 编译成功**（2026-05-29 15:40）
  - 修复 EquipmentDTO/WorkReportDTO/ProcessRouteDTO/ProductionTaskDTO 的 workstationId 字段类型（String → Long）
  - 修复 SopDocument.bindRoute / ProductionTaskCommandService / AndonServiceImpl / MesDomainServiceImpl 等方法参数类型
  - 修复 SopDocumentServiceImpl / ProductionTaskController 的 workstationId 参数类型
  - 共修复 15 处类型不兼容问题

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

- [ ] **设备参数与报工集成** ⚠️ 已预留集成点（2026-05-29）
  - ⚠️ `ProductionTaskCommandService.completeTask()` 已添加 `parameterValuesJson` 参数接收设备参数
  - ⚠️ 预留了 `validateParameterValues()` 方法待实现（需查询 ProcessParameter 仓库获取参数定义并验证）
- [ ] **防错规则集成到任务执行** ⚠️ 已预留集成点（2026-05-29）
  - ⚠️ `ProductionTaskCommandService.completeTask()` 已添加 `performPokayokeCheck()` 方法调用
  - ⚠️ 预留了调用 `PokaYokeService.checkMaterial/checkSequence/checkParameter` 的集成框架
  - ⚠️ 实际防错检查需根据任务上下文（物料、BOM、工序顺序）查询相关规则后执行

### 5.3 工位管理

- [ ] **工艺参数模板**
  - 创建 ParameterGroupTemplate（参数组模板）实体
  - 模板编码、名称、关联产品类型、参数列表
  - DO/Mapper/RepositoryImpl/Controller/DB 表
- [ ] **产品类型到参数组绑定** — 按产品类型自动关联参数组模板

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
- [x] 创建 AccountDO/AccountMapper/AccountRepositoryImpl ✅ 已完成（2026-05-29）
- [x] 创建 AccountSubjectDO/AccountSubjectMapper/AccountSubjectRepositoryImpl ✅ 已完成（2026-05-29）
- [x] 创建 VoucherDO/VoucherMapper/VoucherRepositoryImpl（含 VoucherLine 明细）✅ 已完成（2026-05-29）
- [x] Flyway 迁移：`V1__finance_init.sql`（account、account_subject、voucher、voucher_line 表）✅ 已完成（2026-05-29）
- [ ] 集成测试（20+ 用例覆盖账户操作、凭证审批流）
- [ ] 空 adapter 目录处理：`adapter/grpc/`、`adapter/http/`、`adapter/vo/`（实现或清理）

#### invoice-service（7个文件，端口 10111）

**实体逻辑**：✅ Invoice（129行，DRAFT→ISSUED→PRINTED→VOIDED/RED_FLUSHED 完整状态机）
**Controller**：✅ 完整 CRUD + 开票/打印/作废/红冲

**待补充**：
- [x] 创建 InvoiceDO/InvoiceMapper/InvoiceRepositoryImpl ✅ 已完成（2026-05-29）
- [x] Flyway 迁移：`V1__invoice_init.sql`（invoice 表）✅ 已完成（2026-05-29）
- [ ] 集成测试（10+ 用例覆盖发票全生命周期）

#### settlement-service（12个文件，端口 10113）

**实体逻辑**：✅ SettlementOrder（110行，含自动佣金计算 `settlementAmount = orderAmount - (orderAmount * commissionRate)`）、ReconciliationRecord（95行）、SplitRule（118行，4种分账规则：比例/固定/保底封顶/混合）
**定时任务**：✅ 日自动结算 (`0 0 2 * * ?`) + 月对账触发 (`0 0 0 1 * ?`)
**Controller**：✅ 完整 CRUD + 确认/处理/完成/失败/取消/退款

**待补充**：
- [x] 创建 SettlementOrderDO/SettlementOrderMapper/SettlementOrderRepositoryImpl ✅ 已完成（2026-05-29）
- [ ] 创建 ReconciliationRecordDO/ReconciliationRecordMapper/ReconciliationRecordRepositoryImpl
- [ ] 创建 SplitRuleDO/SplitRuleMapper/SplitRuleRepositoryImpl
- [x] Flyway 迁移（settlement_order、reconciliation_record、split_rule 表）✅ 已完成（2026-05-29）
- [ ] 集成测试（15+ 用例覆盖结算流程、分账计算）

#### reporting-service（已在第 4 节详细列出）
- [ ] 按第 4 节计划完整重写（实体 ORM + DO/Mapper + RepositoryImpl + Flyway + 真实聚合 + 测试）

---

### 7.2 供应链域 — 持久化层不完整

| 服务 | 文件数 | 持久化状态 | 紧急程度 |
|------|--------|-----------|:-------:|
| **warehouse-service**（10106） | 33 | ✅ **唯一完整**（3 DO + 3 Mapper + 2 Converter + 2 RepositoryImpl） | 低 |
| **inventory-service**（10105） | 18 | ✅ **持久化已完成**（DO + Mapper + Converter + RepositoryImpl + schema.sql，2026-05-29） | 低 |
| **logistics-service**（10107） | 15 | ✅ **持久化已完成**（DO + Mapper + Converter + RepositoryImpl + schema.sql，2026-05-29） | 低 |
| **supplier-service**（10109） | 13 | ✅ **持久化已完成**（DO + Mapper + RepositoryImpl + schema.sql，2026-05-29） | 低 |
| **procurement-service**（10108） | 13 | ✅ **持久化已完成**（DO + Mapper + RepositoryImpl + schema.sql，2026-05-29） | 低 |

#### warehouse-service（补充完善）
- [ ] Flyway 迁移：`V1__warehouse_init.sql`（warehouse、location、inbound_order、inbound_order_item、outbound_order、outbound_order_item 表）
- [ ] 单元测试 + 集成测试（覆盖入库/出库完整流程）

#### inventory-service（需完整建持久层）✅ 已完成（2026-05-29）
- [ ] 集成测试（覆盖预留/确认/取消/增减库存流程）

#### logistics-service（需完整建持久层）✅ 已完成（2026-05-29）
- [ ] 集成测试

#### supplier-service（替换内存存储）✅ 已完成（2026-05-29 17:00）
- [x] 创建 SupplierDO/SupplierMapper/SupplierRepositoryImpl（替换 ConcurrentHashMap）
- [x] 添加 Controller 权限注解（`@RequirePermission`）- 已存在
- [x] Flyway 迁移：`schema.sql`（supplier 表）
- [ ] 集成测试

#### procurement-service（替换内存存储）✅ 已完成（2026-05-29 17:00）
- [x] 创建 ProcurementOrderDO/ProcurementOrderMapper/ProcurementOrderRepositoryImpl
- [x] 添加 Controller 权限注解（`@RequirePermission`）- 已存在
- [x] Flyway 迁移：`schema.sql`（procurement_order 表）
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

- [ ] **API 封装** (`src/apis/qc.ts`) ✅ 已修复（2026-05-29）
  - 问题1：多个 API 函数 URL 路径为空 → 已补全所有 URL 路径（30+ 函数）
  - 问题2：部分函数缺少参数传递 → 已修复参数传递
  - 问题3：type-check 有 30+ 错误，`http.get/post/put/delete` 方法不存在于 http 函数 → 已修复为 `http({method:'get',url:...})` 格式（SPC 模块 10 个函数）
  - 问题4：DefectSeverity 类型中 `' observation'` 有前导空格（typo）→ 已修正
  - 构建验证：✅ npm run build-only 成功
  - type-check 验证：✅ vue-tsc --noEmit 通过（0 错误）
- [ ] **不合格处置页** (`src/views/mes/qc/nonConformance/`) ✅ 已完成（2026-05-29）
- [ ] **Store + 路由 + 国际化** ✅ 已完成（2026-05-29）
### 8.2 ProductionTask 生产任务

**后端**：✅ 完整（112行实体 + 237行 Controller + CommandService + QueryService + RepositoryImpl）
**前端**：✅ 已完成（2026-05-29）

- [ ] **API 封装** (`src/apis/productionTask.ts`) ✅ 已完成
- [ ] **生产任务列表页** (`src/views/mes/productionTask/index.vue`) ✅ 已完成
- [ ] **生产任务详情页** (`src/views/mes/productionTask/detail.vue`) ✅ 已完成
- [ ] **生产任务表单页** (`src/views/mes/productionTask/form.vue`) ✅ 已完成
- [ ] **路由 + 国际化** ✅ 已完成

### 8.3 WorkOrder 工单管理

**后端**：✅ 完整（296行实体，含完整状态机 + 拆分规则 + 物料需求计算）
**前端**：✅ 已完成（2026-05-29）

- [ ] **API 封装** (`src/apis/workOrder.ts`) ✅ 已存在
- [ ] **工单列表页** (`src/views/mes/workOrder/index.vue`) ✅ 已完成
- [ ] **工单详情页** (`src/views/mes/workOrder/detail.vue`) ✅ 已完成
- [ ] **工单表单页** (`src/views/mes/workOrder/form.vue`) ✅ 已完成
- [ ] **Store + 路由 + 国际化** ✅ 已完成

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

### 11.1 Supplier/Procurement 控制器缺少权限注解 ✅ 已完成（2026-05-29）

**问题**：supplier-service 和 procurement-service 的 Controller 未使用 `@RequirePermission` 注解，与项目中其他服务不一致。

**修复状态**：经核查，两个 Controller 均已正确使用 `@RequirePermission` 注解，无需修改。

### 11.2 InventoryDomainServiceImpl.findBySkuAndWarehouse 始终返回空 ✅ 已完成（2026-05-29）

**文件**：`inventory-service/.../InventoryDomainServiceImpl.java`
**问题**：方法体为 `return Optional.empty()`，未实现实际查询逻辑
**修复状态**：经核查，该方法已正确调用 repository.findBySkuAndWarehouse() 实现真实查询，repository 实现也已正确使用 MyBatis-Plus 查询。

### 11.3 LogisticsApplicationServiceImpl.listOrders 返回空列表 ✅ 已修复（2026-05-29）

**文件**：`logistics-service/.../LogisticsApplicationServiceImpl.java`
**问题**：直接返回 `List.of()`
**修复**：已实现真实查询逻辑，根据 carrierId 和 status 参数查询物流订单
# 代码规范审查结果 (2026-05-29 16:00)

## 审查概要

本次审查验证了以下已完成项目是否满足代码规范要求：

### 编译验证
- ✅ 前端构建成功 (npm run build-only)
- ✅ Type-check 通过 (vue-tsc --noEmit)

### 代码规范验证
- ✅ MES 模块 catch 块使用 ElMessage.error（无 console.error）
- ✅ MES 模块路由使用国际化键 (t() 函数)
- ✅ 设备表单页 catch 块使用 ElMessage.error
- ✅ 角色列表页 catch 块使用 ElMessage.error
- ✅ 菜单列表页 catch 块使用 ElMessage.error

### 审查结论
经审查，以上已完成项目均符合代码规范要求（CODE_PINCEPLES/CODE_PRICEPLES、CODE_PINCEPLES/FRONTEND_PRICEPLES、CODE_PINCEPLES/CHECK_RULE.md）。