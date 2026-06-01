# TODO

准则：代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)，所有代码中的文本（注释、日志、变量命名等）统一使用英文，国际化文本除外
添加后端服务与功能后，需要考虑是否需要添加浏览与操作到[后台管理](apps/backstage-admin/)或[数字孪生](apps/digital-twin/)里

---
**【待规划功能】（2026-06-01评估）**

- [ ] **智能客服系统** — AI对话式客服、常见问题自动回复、工单智能分类；与订单、售后服务集成；自然语言处理
- [ ] **直播带货系统** — 主播开播、商品挂载、实时弹幕、订单生成；与订单、支付服务集成
- [ ] **社交电商系统** — 拼团、分享奖励、裂变分销、社群管理；与用户、订单服务集成

---
**【项目演进任务】（2026-06-01）**

~~社交电商系统架构设计~~：编译错误已修复：
1. group-buying-service/GroupBuyApplicationService.java 第90行：LocalDateTime 转换为 Timestamp 后传给 expireTime 字段；第197行：Timestamp 与 Timestamp.valueOf(LocalDateTime.now()) 比较
2. social-commerce-service/DistributionRewardDO.java 第17行：补上缺失的 @ 符号（AllArgsConstructor → @AllArgsConstructor）
编译通过，已从清单中移除

---
**【服务合并】：**

- [ ] **合并 notification-service 到 message-service** -- 代码规范问题：缺少单元测试，不符合CODE_PRICEPLES核心业务逻辑与底层能力必须有单元测试要求。message-service 已整合多渠道发送能力（短信、邮件、App推送、站内通知、模板系统），编译通过，但需补充 NotificationApplicationService 等核心类的单元测试

---
**【前端管理后台缺失】：** 

以下后端服务已创建，但 `apps/backstage-admin/` 和 `apps/digital-twin/` 中缺少对应的管理和操作页面，需逐项补充：

- mall-domain（11 个服务中大部分缺少后台管理页面）
- ai-domain（4 个服务）
- factory-domain / mes-service（生产管理后台）
- blockchain-domain（2 个服务）
- erp-domain（6 个服务：财务、HR、发票、项目、报表、结算）
- platform-domain（7 个服务：佣金、客服、数据分析、媒体、消息、通知、用户行为）
- supply-chain-domain（6 个服务：库存预警、库存、物流、采购、供应商、仓库）

---
**【代码规范审查结论】（2026-06-01）**

本次审查处理结果：
- 智能客服系统架构设计：已完成代码规范检查和编译验证通过（WorkOrder、Faq领域模型设计，WorkOrderRepository、FaqRepository接口，WorkOrderService、FaqService、TicketClassificationService应用服务，AI工单智能分类实现），已通过检查并从清单中移除
- 供应商管理系统：artifactId 修复（mall-supplier-service）已通过检查并从清单中移除
- 营销活动中心：PromotionException 继承 BusinessException 已通过检查并从清单中移除
- 链上商品溯源：使用 OpenZeppelin initializer 修饰符，符合区块链合约安全最佳实践，已通过检查并从清单中移除
- 编译阻塞已修复：OrderPaymentSaga.java 移除 @Data 注解，order-service 编译成功，已通过检查并从清单中移除
- mall-supplier-service服务目录名：已通过检查并从清单中移除
- 推荐服务架构迁移：编译通过，代码规范问题已修复，已通过检查并从清单中移除
- 智能推荐系统：代码已实现基于配置的真算法逻辑，编译通过，已通过检查并从清单中移除
- 会员等级系统：编译通过，已通过检查并从清单中移除
- 运营数据看板：编译通过，已通过检查并从清单中移除
- 财务对账自动化：编译通过，已通过检查并从清单中移除
- 智能风控系统：编译通过，已通过检查并从清单中移除
- Solidity 编译错误为预存问题（GoodsNFT.sol、CommitReveal.sol），非本次修改引起，保留在修正需求中
- 售后服务系统、商品评价系统、库存预警系统、智能客服系统：编译通过，代码规范检查已通过，从清单中移除
- 消息通知中心：代码已实现真实 RPC 调用和短信网关集成，编译通过，代码规范问题已修复，已从清单中移除
- 库存预警系统管理后台：编译通过，代码实现完整，已通过检查并从清单中移除
- 推荐系统管理后台：编译通过，代码实现完整（deleteRuleAPI已实现，handleDelete调用真实API，后端deleteRule端点存在），已通过检查并从清单中移除
- 前端项目编译错误已修复（http.ts 中 http.get/post/put/delete 方法类型定义完整，index.vue 中 TypeScript 类型错误已修正），编译通过，已通过检查并从清单中移除
- 直播带货系统架构设计：LiveApplicationService 方法已拆分为私有辅助方法，符合单函数不超过20行规范；已新增 LiveApplicationServiceTest 单元测试类（9个测试用例覆盖核心业务逻辑）；编译通过，已通过检查并从清单中移除
- 社交电商系统架构设计：编译错误，group-buying-service 和 social-commerce-service 存在类型不兼容和语法错误，标记为未完成

**备注**：
- 推荐系统管理后台：已从清单中移除，代码符合规范要求
- 库存预警系统管理后台：已从清单中移除，代码符合规范要求
- 前端项目编译错误已修复，编译通过
- 直播带货系统架构设计：已从清单中移除，代码符合规范要求
- 社交电商系统架构设计：标记为未完成，需修复编译错误