# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 待完成任务

- [ ] **固定资产模块**：资产卡片、折旧计提（直线/双倍余额/年数总和）、资产盘点、资产减少/报废处置
  - 问题：后端实体代码（server/erp-domain/ 下不存在 FixedAsset 相关 Java 文件）和前端页面（apps/backstage-admin/src/views/ 下不存在 asset 相关目录）均不存在，属于未实现功能

- [ ] **资金管理**：资金计划、现金流预测、银行对账、资金调拨
  - 问题：后端代码存在但有编译错误
    - CashFlowDirection 类找不到 (CashPlan.java:143) ✅ 已修复（CashPlanLine 内部已定义，修复引用方式）
    - CashConverter 转换器类找不到 (BankAccountRepositoryImpl.java) ✅ 已创建

- [ ] **多币种核算**：支持外币凭证、汇兑损益处理
- [ ] **供应商绩效评估**：交货及时率、质量合格率、价格竞争力评分模型及看板
- [ ] **库存ABC分类管理**：按存货价值/周转率划分A/B/C类，制定不同管理策略
- [ ] **自动补货建议**：基于安全库存、历史销量、采购提前期自动生成请购单
- [ ] **需求预测**：基于时间序列或简单移动平均的销量预测
- [ ] **仓库质检记录**：独立质检标准表、质检结果（合格/不合格/让步接收）、不良品处理流程
- [ ] **仓库作业策略**：配置先进先出/后进先出/指定批次出库规则
- [ ] **物流费用自动结算**：物流服务与结算服务联动，自动生成运费结算单
- [ ] **供应商协同门户**：供应商自助查询订单、发货通知、对账
- [ ] **高级成本会计**：作业成本法（ABC）、标准成本与实际成本差异分析
- [ ] **财务比率仪表盘**：库存周转率、应收账款周转天数、应付账款周转天数、毛利率等关键指标可视化
- [ ] **报表订阅与自动发送**：定时生成报表并通过邮件/钉钉发送
- [ ] **人力资源（HRM）模块**：员工档案、组织架构、薪资核算、考勤管理（可选，后期规划）
- [ ] **项目管理模块**：项目预算、任务分解、工时填报、成本归集（可选，后期规划）

## 编译错误修复（新增）

- [x] **finance-service 编译错误修复** ✅ 已通过审查：Maven 编译通过
  - CashCommandService.java: 修复了 updateCashPlan 方法使用 findByPlanCode 替代 findById，添加了缺失的 Repository 方法 (update, findByFromAccountIdOrToAccountId, saveItem, findItemsByForecastId)
  - CashTransferRepository/BankReconciliationRepository/CashFlowForecastRepository: save 方法返回类型改为 Long
  - BankReconciliation.java: 修复了枚举引用 (ReconciliationItemStatus → ReconciliationItem.ReconciliationItemStatus)
  - CashQueryService.java: 修复了枚举引用 (BankReconciliationStatus → ReconciliationStatus)
  - CashPlanCreateCommand.java: 将内部类改为 public static class

- [x] **tsconfig.json 弃用选项** ✅ 已通过审查：apps/digital-twin/system-management/tsconfig.json 已添加 "ignoreDeprecations": "6.0"，前端编译通过

## 已完成任务（审查后移除）

- ✅ 预算管理：预算编制、执行控制、预算调整、预算与实际对比分析（后端实体/服务/Controller + 前端页面已完整实现，编译通过）
- ✅ tsconfig.json 配置：moduleResolution 已从 node10 修改为 Node，编译通过
- ✅ 前端 TypeScript 编译错误修复：测试文件和生产代码类型错误已全部修复，vue-tsc 编译通过
- ✅ tsconfig.json 弃用选项：已添加 ignoreDeprecations 配置，前端编译通过

- [ ] 除了国际化文本和[本文档](TODO.md)可以包含中文外，其他所有文本内容都使用纯英文