# TODO

准则：代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)，所有代码中的文本（注释、日志、变量命名等）统一使用英文，国际化文本除外

- [ ] **供应商管理系统** — 供应商入驻、资质审核、绩效评估、采购结算；创建 supplier-service；使用自定义业务异常 BusinessException 替代 RuntimeException；与采购服务集成（ProcurementService RPC）；编译通过
  - ✅ 已修复：mall-domain/supplier-service 的 artifactId 改为 mall-supplier-service，避免重复定义
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起：OrderPaymentSaga.java 使用 @Data 注解 Lombok）
- [ ] **智能推荐系统** — 创建 recommendation-service (product-recommendation-service)；实现基于协同过滤、内容推荐和混合推荐的算法；与用户行为数据、商品画像集成；编译通过
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）
- [ ] **会员等级系统** — 实现用户会员等级、积分规则、等级权益；MemberLevelService 已实现；RPC 代码已生成；编译通过
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）
- [ ] **营销活动中心** — 秒杀、团购、限时折扣活动；promotion-service 已实现活动配置、限时秒杀、团购活动；与库存、订单服务集成
  - ✅ 已修复：PromotionException 继承 BusinessException（替代 RuntimeException）
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）
- [ ] **链上商品溯源** — 商品全链路区块链溯源；合约使用 AccessControl 角色控制替代单一 EOA owner；后端使用自定义业务异常 BusinessException 替代 RuntimeException；编译通过
  - ✅ 已修复：ProductTraceability.sol 使用 TimelockController 替代 msg.sender 作为 admin
  - ✅ 已添加：initialize() 函数，需通过 TimelockController (多签) 初始化合约
  - ✅ 已添加：TraceCreated 事件声明
  - ❌ 未通过代码规范检查（BLOCK_CHAIN_CONTRACT_PRICEPLES-Upgrade/Proxy）：initialize() 函数仍使用自定义 `bool public initialized` 变量，缺少 OpenZeppelin 的 `initializer` 修饰符，不符合区块链合约安全最佳实践
  - ❌ 修复建议未执行：需使用 OpenZeppelin 的 initializer 修饰符替代自定义 initialized 变量
    1. 删除 `bool public initialized;` 变量声明
    2. 将 `function initialize(address _timelock) external {` 改为 `function initialize(address _timelock) external initializer {`
    3. 删除函数体内的 `require(!initialized, "Already initialized");` 和 `initialized = true;` 行
  - ❌ 未通过编译检查：整体编译失败（编译错误非本次修改引起：OrderPaymentSaga.java 使用 @Data 注解 Lombok）
- [ ] **运营数据看板** — 销售、用户、库存多维统计分析；DTO 类拆分文件；移除 Lombok 依赖；data-analysis-service 编译通过
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）
- [ ] **财务对账自动化** — 日终自动生成财务对账报表，检测账务异常；代码无中文注释；方法拆分合理；实现 CSV 文件保存逻辑
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）
- [ ] **智能风控系统** — 交易风控、反欺诈、异常检测；创建 risk-control-service；实现交易风险评估、异常行为检测；与订单服务集成（OrderClient 使用 @DubboReference）；编译通过
  - ❌ 未通过编译检查：待整体编译验证（编译错误非本次修改引起）

---
**【修正】整体编译错误记录（非本次修改引起）：**
- ❌ 编译失败：order-service 中 OrderPaymentSaga.java 使用 @Data 注解（Lombok），但项目已移除 Lombok 依赖
  - 影响范围：9 处 cannot find symbol: class Data
  - 状态：已记录在 TODO 中，待修复

---
**【代码规范审查结论】（2026-06-01）**

本次审查处理结果：
- ✅ 已通过代码规范检查并从 TODO.md 删除的项目：
  - 代码审查修复：payment-service（RuntimeException → BusinessException）
  - 代码审查修复：order-service（中文注释）（已删除中文注释）
  - 代码审查修复：order-service（语法错误）（RuntimeException → BusinessException）
  - 代码审查修复：user-service（RuntimeException → BusinessException + 英文错误信息）
  - 服务间调用统一使用 Dubbo RPC（RestTemplate → @DubboReference）
  - 营销活动中心（PromotionException → BusinessException）
  - 供应商管理系统（artifactId 修复）

- ❌ 链上商品溯源：未通过代码规范检查（BLOCK_CHAIN_CONTRACT_PRICEPLES-Upgrade/Proxy）
  - 问题：ProductTraceability.sol 的 initialize() 函数使用自定义 `bool public initialized` 变量
  - 违反规范：缺少 OpenZeppelin 的 `initializer` 修饰符，不符合区块链合约安全最佳实践
  - 修复方案：
    1. 删除 `bool public initialized;` 变量声明
    2. 将 `function initialize(address _timelock) external {` 改为 `function initialize(address _timelock) external initializer {`
    3. 删除函数体内的 `require(!initialized, "Already initialized");` 和 `initialized = true;` 行

- ❌ 编译阻塞：OrderPaymentSaga.java 使用 @Data 注解（Lombok），项目已移除 Lombok 依赖，导致整体编译失败
  - 状态：已记录在修正项中，待修复后重新验证
- [] 所有<spring-boot-starter-web mybatis-plus-spring-boot3-starter org.projectlombok 都已经写在 @server/pom.xml 父项目里并且被微服务继承，不需要重复在微服务的pom.xml里写这些依赖，需要检查并移除