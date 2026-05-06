# 电商App功能补全 TODO

## 项目状态
- 框架：Expo SDK 54 + React Native 0.81.5
- 路由：expo-router ~6.0.23
- 语言：TypeScript 5.9
- 状态管理：React Context + Custom Hooks

## 核心购物链路现状
✅ 推荐商品 → ✅ 广告 → ✅ 商品列表 → ✅ 商品详情 → ✅ 规格选择 → ✅ 加入购物车 → ✅ 注册登录 → ✅ 商品结算 → ✅ 收货地址 → ✅ 订单信息 → ✅ 物流信息 → ✅ 确认收货/退货

---

## 后端接口补全（对比 temp/mall 缺失项）

### 16. 用户模块接口补全
- [x] 16.1 `GET /sso/getAuthCode` - 获取短信验证码
- [x] 16.2 `POST /sso/updatePassword` - 修改密码
- [x] 16.3 `GET /sso/refreshToken` - 刷新 Token

### 17. 搜索模块接口补全（整个模块缺失）
- [x] 17.1 `GET /v1/products/search` - 综合搜索（替代 ES）
- [x] 17.2 `GET /v1/products/search/simple` - 简单搜索
- [x] 17.3 `GET /v1/products/recommend/{id}` - 商品推荐
- [ ] 17.4 `POST /esProduct/importAll` - 导入商品到 ES（待 ES 服务）
- [ ] 17.5 `POST /esProduct/create/{id}` - 同步商品到 ES（待 ES 服务）

> **参考**: `temp/mall/mall-search/src/main/java/com/macro/mall/search/controller/EsProductController.java`

### 18. 首页模块接口补全
- [x] 18.1 `GET /v1/home/productCateList` - 首页商品分类
- [x] 18.2 `GET /v1/home/hotProductList` - 人气推荐商品
- [x] 18.3 `GET /v1/home/newProductList` - 新品推荐商品
- [ ] 18.4 `GET /home/subjectList` - 首页专题内容

### 19. 购物车模块接口补全
- [x] 19.1 `GET /v1/cart/list/promotion` - 购物车促销信息
- [x] 19.2 `POST /v1/cart/update/attr` - 修改购物车商品规格
- [x] 19.3 `GET /v1/cart/getProduct/{productId}` - 获取商品规格

### 20. 优惠券模块接口补全
- [x] 20.1 `GET /v1/promotion/coupons/listByCart` - 购物车相关优惠券

### 21. 浏览历史模块（整个模块缺失）
- [x] 21.1 `POST /v1/read-history/create` - 创建浏览记录
- [x] 21.2 `GET /v1/read-history/list` - 浏览记录列表
- [x] 21.3 `DELETE /v1/read-history` - 删除浏览记录
- [x] 21.4 `DELETE /v1/read-history/clear` - 清空浏览记录

### 22. 收藏模块（整个模块缺失）
- [x] 22.1 `POST /v1/product-collections/add` - 添加收藏
- [x] 22.2 `GET /v1/product-collections/list` - 收藏列表
- [x] 22.3 `DELETE /v1/product-collections` - 删除收藏
- [ ] 22.4 `GET /v1/product-collections/detail` - 收藏详情
- [x] 22.5 `DELETE /v1/product-collections/clear` - 清空收藏

### 23. 品牌关注模块（整个模块缺失）
- [x] 23.1 `POST /v1/member-attentions/add` - 添加品牌关注
- [x] 23.2 `GET /v1/member-attentions/list` - 关注列表
- [x] 23.3 `DELETE /v1/member-attentions` - 取消关注
- [ ] 23.4 `GET /v1/member-attentions/detail` - 关注详情
- [x] 23.5 `DELETE /v1/member-attentions/clear` - 清空关注

### 24. 订单模块接口补全
- [x] 24.1 `POST /order/cancelTimeOutOrder` - 自动取消超时订单

### 25. 支付宝支付流程补全
- [x] 25.1 `POST /pay/alipay/callback` - 支付异步回调（已整合到 PayController）
- [x] 25.2 `GET /pay/alipay/query` - 支付宝支付状态查询

---

### 缺失接口补全状态
- [x] `/pay/alipay/query` - 支付宝支付状态查询
- [x] `GET /v1/addresses/{id}` - 单个地址详情
- [x] `GET /v1/member-attentions/detail` - 品牌关注详情（后端已实现）
- [x] `GET /v1/product-collections/detail` - 商品收藏详情（后端已实现）
- [x] `GET /v1/brands/recommendList` - 品牌推荐列表

### 后端待实现接口
- [ ] `POST /esProduct/importAll` - 导入商品到 ES（待 ES 服务）
- [ ] `POST /esProduct/create/{id}` - 同步商品到 ES（待 ES 服务）
- [ ] `GET /home/subjectList` - 首页专题内容（返回空列表，需完善后端实现）
- [ ] `GET /order/{id}/logistics` - 物流信息查询（需新增）
- [ ] `POST /order/{id}/refund` - 退款申请（需新增）
- [ ] `GET /order/{id}/refund/status` - 退款状态查询（需新增）

### 前端待完善功能
- [x] `app/category/[id].tsx` - 商品分类列表页（已完成）
- [x] `app/favorites.tsx` - 收藏列表页改用后端 API（已完成）
- [x] `app/orders/index.tsx` - 订单列表分页（前端已完成）
- [ ] `app/orders/[id]/logistics.tsx` - 物流信息页（使用 MOCK 数据，需接入真实 API）
- [ ] `app/orders/[id]/refund.tsx` - 退款申请页（使用 MOCK 数据，需接入真实 API）
- [ ] `app/orders/[id]/review.tsx` - 评价页面（需完善）
- [ ] `app/notifications.tsx` - 消息中心（需完善未读徽标）
- [ ] `app/coupons.tsx` - 优惠券列表（需完善领取功能）

---

## P0 - 核心功能（必须完成）

### 1. 用户注册/登录页面
- [x] 1.1 创建登录页面 `app/auth/login.tsx`
- [x] 1.2 创建注册页面 `app/auth/register.tsx`
- [x] 1.3 邮箱密码登录
- [x] 1.4 用户名密码登录（SSO `/sso/login` 接口）
- [x] 1.5 手机号+验证码登录
- [ ] 1.6 第三方登录（微信/Apple）
- [x] 1.7 忘记密码页面
- [x] 1.8 完善 AuthContext 集成（含 SSO 登录/注册）

### 2. 商品搜索页面
- [x] 2.1 创建搜索页面 `app/search.tsx`
- [x] 2.2 搜索历史记录
- [x] 2.3 热门搜索推荐
- [x] 2.4 搜索结果页（分页加载）
- [ ] 2.5 搜索结果筛选/排序（待后端支持）
- [x] 2.6 首页搜索框跳转

### 3. 商品列表页
- [x] 3.1 创建分类商品列表页 `app/category/[id].tsx`
- [x] 3.2 商品列表展示（网格/列表视图切换）
- [x] 3.3 分页加载（待后端分页支持）
- [x] 3.4 筛选功能（价格区间，待后端支持）
- [x] 3.5 排序功能（综合/销量/价格）

### 4. 商品规格/SKU选择
- [x] 4.1 创建规格选择弹窗组件 `components/product/SKUSelector.tsx`
- [x] 4.2 SKU数据获取和展示
- [x] 4.3 规格组合选择逻辑
- [x] 4.4 库存联动显示
- [x] 4.5 价格联动显示
- [x] 4.6 集成到商品详情页

### 5. 收货地址管理
- [x] 5.1 创建地址列表页 `app/address/list.tsx`
- [x] 5.2 创建地址编辑页 `app/address/edit.tsx`
- [x] 5.3 新增地址表单
- [x] 5.4 编辑地址表单
- [x] 5.5 删除地址
- [x] 5.6 设置默认地址
- [x] 5.7 结算页地址选择入口

### 6. 订单列表页
- [x] 6.1 创建订单列表页 `app/orders/index.tsx`
- [x] 6.2 订单状态Tab（全部/待付款/待发货/待收货/已完成/退款）
- [x] 6.3 订单卡片展示
- [ ] 6.4 分页加载（待后端分页支持）
- [x] 6.5 个人中心订单入口集成

### 7. 订单详情页
- [x] 7.1 创建订单详情页 `app/orders/[id].tsx`
- [x] 7.2 订单状态展示
- [x] 7.3 商品信息展示
- [x] 7.4 收货地址信息
- [x] 7.5 支付信息
- [x] 7.6 订单操作按钮（支付/取消/确认收货/退款）

---

## P1 - 重要功能（增强体验）

### 8. 物流信息
- [x] 8.1 创建物流信息页 `app/orders/[id]/logistics.tsx`
- [x] 8.2 物流轨迹展示
- [x] 8.3 物流公司信息
- [x] 8.4 订单详情页物流入口
- [ ] 8.5 物流状态实时更新（待后端支持）

### 9. 确认收货
- [x] 9.1 确认收货功能
- [x] 9.2 收货确认弹窗
- [ ] 9.3 收货后订单状态更新（待后端API）
- [x] 9.4 收货成功提示

### 10. 退货/退款申请
- [x] 10.1 创建退款申请页 `app/orders/[id]/refund.tsx`
- [x] 10.2 退款类型选择（退货退款/仅退款）
- [x] 10.3 退款原因选择
- [x] 10.4 退款金额填写
- [x] 10.5 退款凭证上传
- [x] 10.6 退款进度查询
- [x] 10.7 退款详情页

### 11. 收藏功能
- [x] 11.1 收藏/取消收藏API集成
- [x] 11.2 收藏列表页 `app/favorites.tsx`
- [x] 11.3 商品详情页收藏按钮
- [x] 11.4 个人中心收藏入口

---

## P2 - 次要功能（锦上添花）

### 12. 优惠券系统
- [x] 12.1 优惠券列表页
- [x] 12.2 领取优惠券（后端已有接口）
- [x] 12.3 结算页优惠券选择
- [x] 12.4 优惠券使用规则
- [x] 12.5 个人中心优惠券入口

### 13. 评论功能
- [x] 13.1 商品评论列表（商品详情页预览）
- [x] 13.2 发布评论
- [x] 13.3 评论图片上传
- [x] 13.4 评论点赞
- [x] 13.5 订单完成后评价入口

### 14. 消息通知
- [x] 14.1 消息中心页面
- [x] 14.2 系统通知
- [x] 14.3 订单状态推送
- [x] 14.4 促销活动推送

### 15. 扫码功能
- [x] 15.1 集成 scanner-module
- [x] 15.2 扫码页面
- [x] 15.3 扫码跳转商品详情
- [x] 15.4 首页扫码入口

---

## 当前进行中任务

### Phase 1: 用户认证完善（P0-1.5, 1.7） ✅ 已完成
- [x] 1.5.1 后端新增 `POST /sso/loginByPhone` 接口（参考 temp/mall 的 UmsMemberService）
- [x] 1.5.2 AuthContext 实现 `loginWithPhone` 方法
- [x] 1.5.3 AuthContext 实现 `getAuthCode` 方法调用后端 `/sso/getAuthCode`
- [x] 1.5.4 AuthContext 实现 `forgotPassword` 方法调用后端 `/sso/updatePassword`
- [x] 1.5.5 登录页 `login.tsx` 对接手机验证码登录逻辑
- [x] 1.7.1 创建忘记密码页面 `app/auth/forgot-password.tsx`（参考 temp/mall-app-web 登录页忘记密码入口）

### Phase 2: 结算页地址集成（P0-5.7） ✅ 已完成
- [x] 5.7.1 `checkout.tsx` 添加收货地址区块（参考 temp/mall-app-web/pages/order/createOrder.vue）
- [x] 5.7.2 地址选择跳转到 `/address/list` 并返回选中地址
- [x] 5.7.3 订单创建时传入 `memberReceiveAddressId`

### Phase 3: 商品详情页收藏按钮集成（P1-11.3） ✅ 已完成
- [x] 11.3.1 `api/generated.ts` 添加 `ProductCollectionApi` 类
- [x] 11.3.2 商品详情页 `product/[id].tsx` 添加收藏状态加载
- [x] 11.3.3 收藏按钮点击切换收藏/取消收藏
- [x] 11.3.4 收藏图标动态切换（空心/实心 + 颜色变化）

### Phase 4: 退款凭证上传（P1-10.5） ✅ 已完成
- [x] 10.5.1 安装 `expo-image-picker` 依赖
- [x] 10.5.2 `refund.tsx` 添加图片选择功能
- [x] 10.5.3 图片预览与删除功能
- [x] 10.5.4 最多 5 张图片限制

### Phase 5: 优惠券系统（P2-12） ✅ 已完成
- [x] 12.1 优惠券列表页 `app/coupons.tsx`
- [x] 12.2 `api/generated.ts` 添加 `CouponApi` 类
- [x] 12.3 结算页优惠券选择（Modal 弹窗 + 金额计算 + 订单创建时传入 couponId）
- [x] 12.4 优惠券使用规则展示
- [x] 12.5 个人中心优惠券入口（待集成到个人中心）
- [x] 12.6 后端 `OrderCreateRequest` 新增 `couponId`/`memberReceiveAddressId`/`useIntegration` 字段

### Phase 6: 评论功能（P2-13） ✅ 已完成
- [x] 13.1 后端评论接口补全（`user-action-service` 新增 `ProductComment` 模块）
- [x] 13.2 商品详情页评论预览（`product/[id].tsx` 添加 `CommentPreview` 组件）
- [x] 13.3 发布评论功能（`app/orders/[id]/review.tsx` 创建评价页面）
- [x] 13.4 评论图片上传（集成 `expo-image-picker`，支持最多5张图片）
- [x] 13.5 订单完成后评价入口（订单详情页添加评价按钮，状态3显示）

### Phase 7: 消息通知（P2-14） ✅ 已完成
- [x] 14.1 后端消息模块创建（`message-service` 新增 `Notification` 模块）
- [x] 14.2 消息中心页面（`app/notifications.tsx` 支持分类筛选、已读/删除）
- [x] 14.3 系统通知（`SYSTEM` 类型，支持标题/内容/图标）
- [x] 14.4 订单状态推送（`ORDER` 类型，点击跳转到订单详情）
- [x] 14.5 促销活动推送（`PROMOTION` 类型，支持图片/跳转商品）

### Phase 8: 扫码功能（P2-15） ✅ 已完成
- [x] 15.1.1 `scanner.tsx` 集成 `ScannerModuleView` 和 `ScannerModule`
- [x] 15.1.2 相机权限请求 (`requestCameraPermissionAsync`)
- [x] 15.2.1 扫码状态管理（扫描中/空闲）
- [x] 15.2.2 扫码帧叠加层（四角标记 + 提示文字）
- [x] 15.2.3 错误处理（权限不足/相机错误）
- [x] 15.3.1 扫描结果解析（`product/{id}` 正则匹配）
- [x] 15.3.2 跳转到商品详情页 `/product/[id]`
- [x] 15.4.1 `HomeHeader` 添加扫码图标入口
- [x] 15.4.2 首页导航到 `/scanner`

### Phase 9: 后端接口补全 ⚠️ 进行中
- [x] 9.1 订单自动取消超时 `POST /order/cancelTimeOutOrder` ✅
- [x] 9.2 支付宝回调接口 `POST /pay/alipay/callback` ✅
- [ ] 9.3 首页专题内容 `GET /home/subjectList`（返回空列表，需完善）
- [ ] 9.4 物流信息查询 `GET /order/{id}/logistics`（需新增）
- [ ] 9.5 退款申请接口 `POST /order/{id}/refund`（需新增）
- [ ] 9.6 退款状态查询 `GET /order/{id}/refund/status`（需新增）

### Phase 10: 前端页面完善 ⚠️ 进行中
- [x] 10.1 商品分类列表页 `app/category/[id].tsx` ✅
- [x] 10.2 收藏列表页改用后端 API `app/favorites.tsx` ✅
- [x] 10.3 订单列表页添加分页 `app/orders/index.tsx` ✅
- [ ] 10.4 物流信息页接入真实 API `app/orders/[id]/logistics.tsx`
- [ ] 10.5 退款申请页接入真实 API `app/orders/[id]/refund.tsx`
- [ ] 10.6 个人中心优惠券入口集成

---

## 功能检测清单

### 检测环境要求
- 运行 `yarn install` 完成依赖安装（含 `scanner-module` 本地模块）
- 后端网关运行中，或 `client/api/generated.ts` 已包含手动 API 类
- Expo Dev Server 正常启动（`yarn web` 或 `yarn start`）

---

### Phase 1-2: 用户认证 + 结算地址
- [ ] 1.1 打开登录页，验证用户名密码登录流程
- [ ] 1.2 注册页新增账户，验证入库
- [ ] 1.3 手机验证码登录（需后端 `/sso/getAuthCode` 正常）
- [ ] 1.4 忘记密码页面提交，验证密码更新
- [ ] 1.5 第三方登录（微信/Apple）- **待实现**
- [ ] 2.1 结算页点击地址区块，跳转 `/address/list`
- [ ] 2.2 选中地址返回结算页，显示地址信息
- [ ] 2.3 创建订单后检查 `memberReceiveAddressId` 是否传入

### Phase 3: 商品详情收藏
- [ ] 3.1 未登录状态收藏按钮提示登录
- [ ] 3.2 已登录状态点击收藏，图标变实心
- [ ] 3.3 再次点击取消收藏，图标变空心
- [ ] 3.4 刷新页面后收藏状态保持一致

### Phase 4: 退款凭证上传
- [ ] 4.1 进入退款申请页，点击上传图片
- [ ] 4.2 选择图片后显示预览缩略图
- [ ] 4.3 上传满 5 张图片后隐藏上传按钮
- [ ] 4.4 点击预览图删除按钮可移除图片

### Phase 5: 优惠券系统
- [ ] 5.1 进入优惠券列表页 `/coupons`，展示可用/已用券
- [ ] 5.2 点击领取优惠券，状态变为"已领取"
- [ ] 5.3 结算页点击优惠券区域，弹出选择 Modal
- [ ] 5.4 选择优惠券后显示抵扣金额
- [ ] 5.5 订单创建请求包含 `couponId`
- [ ] 5.6 个人中心菜单可进入优惠券列表

### Phase 6: 评论功能
- [ ] 6.1 商品详情页底部显示评论预览列表
- [ ] 6.2 进入已完成订单，点击"评价"按钮
- [ ] 6.3 评分组件可选 1-5 星
- [ ] 6.4 评论文本输入框正常
- [ ] 6.5 上传图片最多 5 张
- [ ] 6.6 提交评论后刷新详情页可看到新评论
- [ ] 6.7 评论点赞功能正常

### Phase 7: 消息通知
- [ ] 7.1 进入消息中心 `/notifications`，显示三个 Tab（系统/订单/活动）
- [ ] 7.2 未读消息数量徽标显示在个人中心菜单
- [ ] 7.3 点击消息卡片标记为已读
- [ ] 7.4 订单类型消息点击后跳转对应订单详情
- [ ] 7.5 左滑删除消息
- [ ] 7.6 清空已读消息功能

### Phase 8: 扫码功能
- [ ] 8.1 首页右上角扫码图标可点击进入 `/scanner`
- [ ] 8.2 点击"开始扫描"请求相机权限
- [ ] 8.3 授权后摄像头画面正常显示
- [ ] 8.4 扫码框叠加层（四角标记）覆盖在画面上
- [ ] 8.5 识别到 `product/123` 格式二维码后弹出确认框
- [ ] 8.6 点击"查看商品"跳转到对应商品详情页
- [ ] 8.7 识别到非商品二维码显示原始内容
- [ ] 8.8 拒绝相机权限时弹出错误提示
- [ ] 8.9 "停止扫描"按钮可退出扫描模式

---

### 前端待完善页面（需接入真实 API）
- [ ] **物流信息页** `app/orders/[id]/logistics.tsx` - 当前使用 MOCK 数据
  - 需后端新增：`GET /order/{id}/logistics` - 物流信息查询接口
  - 参考：`temp/mall-app-web/pages/order/orderDetail.vue`（仅有入口，无实现）
  
- [ ] **退款申请页** `app/orders/[id]/refund.tsx` - 当前使用 MOCK 数据
  - 需后端新增：`POST /order/{id}/refund` - 退款申请接口
  - 需后端新增：`GET /order/{id}/refund/status` - 退款状态查询接口
  - 参考：`temp/mall-app-web/pages/order/orderDetail.vue`（仅有入口，无实现）

---

### 全局检测
- [ ] G.1 底部 Tab 导航正常（首页/分类/购物车/我的）
- [ ] G.2 深色/浅色模式切换正常
- [ ] G.3 网络错误时显示友好提示
- [ ] G.4 页面返回手势/按钮正常
- [ ] G.5 所有新增路由在 `_sitemap` 中可见
- [x] G.6 `app/category/[id].tsx` 路由已添加
- [x] G.7 `app/favorites.tsx` 集成后端 API
- [x] G.8 `app/orders/index.tsx` 分页功能已添加

---

## 技术债务

### 后端技术债务
- [ ] 统一错误处理（全局异常处理器）
- [ ] 服务间调用优化（RestTemplate → WebClient）
- [ ] 商品 ES 搜索实现（待 ES 服务）
- [ ] 首页专题内容 `GET /home/subjectList` 完善
- [ ] 物流查询接口新增
- [ ] 退款申请接口新增
- [ ] 订单自动取消定时任务（Quartz/Spring Scheduled）
- [ ] 支付宝异步回调完整实现（生产环境配置）

### 前端技术债务
- [ ] 统一错误处理（API 调用错误拦截）
- [ ] 网络请求拦截器优化（Axios 拦截器）
- [ ] 图片缓存优化（react-native-fast-image）
- [ ] 性能优化（FlatList 优化/内存优化）
- [ ] 离线缓存策略（React Query/Apollo Cache）
- [ ] 单元测试补充（Jest/React Native Testing Library）
- [ ] E2E 测试补充（Detox/Appium）
- [x] `api/generated.ts` 使用 OpenAPI 代码生成
- [x] `app/category/[id].tsx` 商品分类列表页已完成
- [x] `app/favorites.tsx` 已改用后端 API

---

## 优先级说明

- **P0**: 核心购物链路必须功能，直接影响用户完成购买
- **P1**: 重要增强功能，影响用户体验和留存
- **P2**: 锦上添花功能，提升App完整度

## 预计完成顺序

1. ~~用户注册/登录页面 (P0-1)~~ ✅
2. ~~商品搜索页面 (P0-2)~~ ✅
3. ~~商品列表页 (P0-3)~~ ✅
4. ~~商品规格/SKU选择 (P0-4)~~ ✅
5. ~~收货地址管理 (P0-5)~~ ✅
6. ~~订单列表页 (P0-6)~~ ✅
7. ~~订单详情页 (P0-7)~~ ✅
8. ~~物流信息页面 (P1-8)~~ ✅（需接入真实API）
9. ~~确认收货 (P1-9)~~ ✅
10. ~~退货/退款申请 (P1-10)~~ ✅（需接入真实API）
11. ~~收藏功能 (P1-11)~~ ✅
12. ~~优惠券系统 (P2-12)~~ ✅
13. ~~评论功能 (P2-13)~~ ✅
14. ~~消息通知 (P2-14)~~ ✅
15. ~~扫码功能 (P2-15)~~ ✅
16. **物流查询接口（后端新增）** ⚠️
17. **退款申请接口（后端新增）** ⚠️
18. **首页专题内容完善** ⚠️
19. **ES 商品搜索（待 ES 服务）** ⚠️
