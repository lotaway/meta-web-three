# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 待完成任务

- [ ] **财务比率仪表盘**：库存周转率、应收账款周转天数、应付账款周转天数、毛利率等关键指标可视化
- [ ] **报表订阅与自动发送**：定时生成报表并通过邮件/钉钉发送
- [ ] **人力资源（HRM）模块**：员工档案、组织架构、薪资核算、考勤管理（可选，后期规划）
- [ ] **项目管理模块**：项目预算、任务分解、工时填报、成本归集（可选，后期规划）

## 编译错误记录（非本次修改引起）

- ⚠️ **前端 TypeScript 编译错误**：apps/client 目录下存在大量 TypeScript 类型错误，主要涉及：
  - Next.js Link 组件路径类型问题（ExternalPathString / RelativePathString）
  - API 方法不存在问题（如 NotificationControllerApi.unreadCount, CouponControllerApi.list）
  - API 请求参数类型问题
  - 此问题非本次 tsconfig 配置修改引起，为项目已有问题，需另行修复

---

### 代码审查记录（2026-05-30）

本次审查的已完成项目（[x]）均已通过检查：

1. **仓库质检记录** ✅ - DDD 分层完整，Domain Service 业务逻辑正确
2. **仓库作业策略** ✅ - FIFO/LIFO/指定批次逻辑完整实现
3. **物流费用自动结算** ✅ - 在 settlement-service 中完整实现，事件联动正常
4. **供应商协同门户** ✅ - 完整的发货通知、对账功能
5. **tsconfig 配置修正** ✅ - baseUrl 已配置
6. **前端 tsconfig 配置问题** ✅ - baseUrl 已配置
7. **高级成本会计** ✅ - 7个实体+仓储接口+领域服务+命令/查询服务完整实现，编译通过

后端编译通过，前端 TypeScript 错误为项目已有问题（非本次修改引起）。