# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端开发规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRINCIPLES) - 前端开发规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）
- [ ] 新建 apps/backstage-admin/.env（gitignored），写入 VITE_BASE_SERVER_URL=http://192.168.1.110:10081（或本机 localhost:10081），避免 axios baseURL 为空导致全部 API 请求失败
- [ ] apps/backstage-admin/src/utils/http.ts SERVICE_PREFIX_MAP 新增 '/api/admin/user': 'mall-user-service'，修复 /api/admin/user/* 无服务前缀导致 404
- [ ] apps/backstage-admin/src/utils/http.ts 将 '/prefrenceArea' 的服务名从 'mall-catalog-service' 改为 'mall-trade-service'，修复错误路由
- [ ] apps/backstage-admin/src/utils/http.ts 将 '/productCategory' 统一为 '/product-category'（短横线），与后端 ProductCategoryController @RequestMapping("/product-category") 对齐
- [ ] apps/backstage-admin/src/apis/productCate.ts 及相关调用处同步将 URL 从 /productCategory/... 改为 /product-category/...
- [ ] apps/client 确认手写 API 的 service 前缀（如 /mall-catalog-service/...）与后端 gateway discovery locator 路由约定一致，端口使用 10081
- [ ] apps/client 运行 openapi-generator 生成 src/generated/api，并确认生成客户端 basePath 带 serviceId 且 baseUrl 指向网关 10081
- [ ] 联调验证：前端指向 http://192.168.1.110:10081 与后端本地配置指向 192.168.1.110 一致，/api/admin/user/*、/product-category/*、/prefrenceArea/* 均能通过网关命中对应服务
