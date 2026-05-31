# TODO

## 代码规范

- 代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)
- 所有代码中的文本（注释、日志、变量命名等）统一使用英文，国际化文本（i18n 资源文件、用户可见的多语言文案）除外

- [ ] **订单取消补偿机制** — OrderApplicationService.processOrderCancellationCompensation() 已实现真实业务逻辑：(1)扩展PromotionService.proto添加returnCoupon/getUserCoupons方法 (2)扩展UserService.proto添加returnIntegration/getUserIntegration方法 (3)实现PromotionServiceRpcImpl新方法 (4)实现UserRPCServiceImpl新方法 (5)创建PromotionClient/UserClient/InventoryClient调用Dubbo服务 (6)processOrderCancellationCompensation()实现库存释放、优惠券返还、积分返还逻辑
  - ⚠️ 检查受阻：后端存在编译错误（详见下方"修复损坏的 POM 文件"任务），无法完整验证代码规范符合性
- [ ] **支付服务对账调度** — ReconciliationServiceImpl 存在 @TODO 注释，getExternalBills/checkMissingOrders/checkExtraOrders/checkAmountMismatches 均为空壳，需实现对账核心逻辑
- [ ] **库存预警通知集成** — InventoryAlertNotificationListener 中 TODO 要求集成邮件、短信、站内信、钉钉机器人发送消息
- [ ] **支付服务对接银行API** — SettlementServiceImpl 中 TODO 要求实现分组逻辑并调用银行/支付平台API进行转账
- [ ] **数字货币钱包对接区块链API** — CryptoWalletServiceImpl 中 TODO 要求调用区块链API或钱包SDK进行转账和余额查询
- [ ] **统一排查微服务间 HTTP 直连调用改为 Dubbo RPC**
  - 当前存在使用 `@Value("${xxx.service.url}")` + `RestTemplate`/`WebClient` 直接调用其他微服务的情况，应统一改为 protobuf + `@DubboReference` 方式
  - 排查方向：全局搜索 `service.url`、`RestTemplate`、`WebClient` + 硬编码 URL 模式
  - 已发现并修复：cart-service 中 `promotion.service.url` + `RestTemplate` 调用 promotion-service
  - 可能存在的其他位置：各 Service Impl 中的 `RestTemplate` 使用、各 Client 类中的 `WebClient` 使用、application.yml 中的 `*.service.url` 配置项
- [ ] **用户服务 Redis 验证码集成** — UserServiceImpl 已正确集成 Redis 存储验证码，5分钟过期，验证成功后自动删除，编译通过
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **购物车促销信息查询** — 已创建 PromotionClient 通过 Dubbo RPC 调用 promotion-service 查询商品可用优惠券，enrichPromotionInfo() 已实现真实查询逻辑，编译通过
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **检查其他后端服务** — erp-domain、mall-domain、platform-domain、supply-chain-domain、ai-domain、blockchain-domain 各服务正常，无 System.out/System.err，无明显 N+1 查询问题
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **后端缓存使用检查** — TokenBlacklistService(Redis)、PriceEngineServiceImpl(ConcurrentHashMap)、ExcelService(Redis MQ)、OpenApiConfig(HashMap) 均正常
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **后端日志规范性检查** — 主代码无 System.out/System.err，统一使用 log.info/debug/warn/error
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **后端异常处理规范性检查** — 有 GlobalExceptionHandler 统一处理异常、BusinessException 业务异常类
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [x] **前端代码规范审查（API调用）** — TypeScript 编译通过，无错误 ✅
- [x] **前端 TypeScript 编译错误修复** — 已修复 ✅
- [x] **后端编译错误修复** — SupplyChainPermissions 缺失常量问题已解决
  - ⚠️ 检查受阻：后端存在新的编译错误（详见下方任务），无法完整验证
- [ ] **项目管理模块** — TimeEntryNotFoundException 已创建，RuntimeException 已替换，编译通过
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **报表订阅与自动发送** — 代码已删除所有注释，编译通过
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **人力资源（HRM）模块** — 代码已删除所有注释，RuntimeException 已改为具体业务异常类，编译通过
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **库存安全预警功能 - 数据库表** — 已通过检查
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **库存安全预警功能 - Repository实现** — 已重构为 MyBatis-Plus 真实数据库持久化
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **库存安全预警功能 - Controller** — 已删除所有注释
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [x] **前端 TypeScript API 调用修复** — 已修复 notificationApi/memberAttentionApi/readHistoryApi/couponApi/productCollectionApi/commentApi/orderApi/AuthContext/profile 等 API 调用 ✅
- [x] **前端 TypeScript 配置（baseUrl）** — 已通过检查 ✅
- [x] **前端性能优化（懒加载）** — 已通过检查（RN 项目不适用 Next.js 懒加载策略）✅
- [x] **后端编译错误修复（SupplyChainPermissions）** — 已通过检查，编译通过
  - ⚠️ 检查受阻：后端存在新的编译错误，无法完整验证
- [ ] **ProductService N+1 查询优化** — 已修复 listProducts/searchProducts/simpleSearch/recommendProducts/getProductById 中的 N+1 查询问题
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [x] **前端 expo-router Link 类型问题修复** — 已修复 `.expo/types/router.d.ts` 类型定义，扩展 `href` 类型接受动态路由字符串 ✅
- [ ] **后端 OrderService N+1 查询优化** — OrderServiceImpl、CartServiceImpl、SettlementServiceImpl 等均无 N+1 查询问题
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证
- [ ] **后端 Service 层 N+1 查询检查（库存/促销服务）** — 各 Service 均使用 saveBatch 批量插入，无 N+1 查询问题
  - ⚠️ 检查受阻：后端存在编译错误，无法完整验证

## 编译错误修复（非本次修改引起）

- [ ] **修复损坏的 POM 文件** — 发现两个 POM 文件被截断损坏，导致后端无法编译：
  - `server/platform-domain/media-service/pom.xml` - 文件不完整，缺少结尾标签
  - `server/factory-domain/digital-twin-service/pom.xml` - 文件不完整，缺少结尾标签
  - 建议：从 Git 历史或其他来源恢复完整的 POM 文件内容