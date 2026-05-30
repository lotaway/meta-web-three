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

- [x] **tsconfig.json 弃用选项** ✅ 已移除（该选项在 TS 5.9 中无效）
  - apps/digital-twin/system-management/tsconfig.json 已移除 ignoreDeprecations 配置
  - 注意：项目仍存在测试文件中的 TypeScript 编译错误（见下方）

- [x] **前端 TypeScript 编译错误修复** 🆕 大部分完成
  - 问题：apps/digital-twin/system-management 项目存在 TypeScript 类型错误
  - 已完成修复：
    - InventoryAlertPanelProps: 已添加 filterLevel、sortBy 属性
    - RestockSuggestionsProps: 已添加 items、onDismiss、filterUrgency 属性，items 现在可作为 suggestions 的别名
    - ShelfHeatmapProps: 已添加 data、title 属性，并改为可选属性
    - WarehouseStatusProps: 已添加 data 属性，并改为可选属性
    - ShelfHeatmap 和 WarehouseStatus 组件代码已更新处理可选属性
    - 修复了 AlertFlow.test.tsx 中 items 变量未定义的 bug
  - 剩余问题：测试文件中 mock 数据类型与接口定义不匹配（11 个错误）
    - AlertFlow.test.tsx: mock 数据缺少必需属性
    - RealTimeDataDisplay.test.tsx: 测试用例使用未定义的 data 和 WarehouseStatusData 变量
    - digital-twin-api.test.ts: Device 类型位置字段类型不匹配 (需要 tuple 而非 object)
  - 建议：修复测试文件中 mock 数据的类型定义

## 已完成任务（审查后移除）

- ✅ 预算管理：预算编制、执行控制、预算调整、预算与实际对比分析（后端实体/服务/Controller + 前端页面已完整实现，编译通过）
- ✅ tsconfig.json 配置：moduleResolution 已从 node10 修改为 Node，编译通过
- ✅ 前端 TypeScript 编译错误修复：测试文件和生产代码类型错误已全部修复，vue-tsc 编译通过
- ✅ finance-service 编译错误修复：Maven 编译通过

- [ ] 除了国际化文本和[本文档](TODO.md)可以包含中文外，其他所有文本内容都使用纯英文