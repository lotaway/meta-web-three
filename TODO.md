# API 迁移待办事项

## 已完成 ✅

### client/api/ (Expo/React Native)
- `home.ts`:
    - `fetchMallHomeContent` - 使用 `brandApi.list()` 获取品牌列表
    - `fetchProductCategoryList` - 使用 `categoryApi.viewChildren()` 获取分类
    - `fetchNewMallProductList` - 使用 `productApi.listProducts({ keyword: 'new' })`
    - `fetchHotMallProductList` - 使用 `productApi.listProducts({ keyword: 'hot' })`
- `product.ts`:
    - `fetchProductDetail` - 使用 `productApi.getProduct()`
    - `fetchProductList` - 使用 `productApi.listProducts()`
- `cart.ts`:
    - `fetchCartList`, `addCartItem`, `deleteCartItems`, `updateCartItemQuantity`, `clearCart` - 使用 `cartApi`
- `order.ts`:
    - `createOrder`, `fetchOrderList`, `fetchOrderDetail`, `cancelOrder`, `confirmReceiveOrder` - 使用 `orderApi`

### client/mall-app-web/api/ (Uni-app)
- `address.js`: 更新为 `/v1/addresses/*` 路径
- `order.js`: 更新为 `/order/*` 路径 (create, list, detail, cancel, confirm-receive)
- `coupon.js`: 更新为 `/v1/promotion/coupons/*` 路径 (list, claim)
- `cart.js`: 更新为 `/v1/cart/*` 路径 (list, add, delete, update/quantity, clear)
- `member.js`: 更新为 `/user/signIn` 路径

### 后端服务迁移 (server/ microservices)
- **首页相关**: 在 `product-service` 中实现了 `HomeController` (`/v1/home/content`, `/v1/home/recommendProductList`)
- **商品分类**: 在 `product-service` 的 `ProductCategoryController` 中实现了 `GET /v1/product-categories/tree` (分类树)
- **品牌相关**: 在 `product-service` 的 `BrandController` 中实现了 `GET /v1/brands/{id}/products` (品牌商品列表)
- **订单相关**: 在 `order-service` 的 `OrderController` 中补全了 `generateConfirmOrder`, `deleteOrder`, `paySuccess`
- **会员互动**: 在 `user-service` 中新增了 `MemberInteractionController` (收藏、足迹、关注) 以及 `UserController.info` (会员信息)
- **优惠券相关**: 在 `promotion-service` 的 `CouponController` 中实现了 `GET /v1/promotion/coupons/listByProduct/{productId}`

---

## 待补充的 API 接口 (需要网关添加)

目前所有在 TODO.md 中列出的核心业务接口均已在对应的微服务中完成迁移和初步实现。

---

## 已知问题

1. **认证逻辑**: 新微服务接口统一依赖 `X-User-ID` header，需要网关或拦截器进行统一处理。
2. **数据填充**: 部分新迁移的接口（如首页聚合内容、会员收藏等）目前为 Mock 实现或部分实现，需要后续接入完整的 Service/Repository 逻辑。
3. **分页参数**: 统一使用 `pageNum`/`pageSize` 或 `page`/`size` 的适配工作仍在进行中。

---

## 迁移建议

1. 启动所有微服务并配置网关路由。
2. 重新运行前端代码生成工具 `npm run generate:api` 以获取最新的 API 定义。
3. 在前端项目中替换所有旧的 API 调用为新的生成代码调用。
