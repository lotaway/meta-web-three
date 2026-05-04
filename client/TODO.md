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
- [x] `GET /v1/member-attentions/detail` - 品牌关注详情
- [x] `GET /v1/product-collections/detail` - 商品收藏详情
- [x] `GET /v1/brands/recommendList` - 品牌推荐列表

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
- [ ] 12.2 领取优惠券（后端已有接口）
- [ ] 12.3 结算页优惠券选择
- [ ] 12.4 优惠券使用规则
- [x] 12.5 个人中心优惠券入口

### 13. 评论功能
- [ ] 13.1 商品评论列表
- [ ] 13.2 发布评论
- [ ] 13.3 评论图片上传
- [ ] 13.4 评论点赞
- [ ] 13.5 订单完成后评价入口

### 14. 消息通知
- [ ] 14.1 消息中心页面
- [ ] 14.2 系统通知
- [ ] 14.3 订单状态推送
- [ ] 14.4 促销活动推送

### 15. 扫码功能
- [ ] 15.1 集成 scanner-module
- [ ] 15.2 扫码页面
- [ ] 15.3 扫码跳转商品详情
- [ ] 15.4 首页扫码入口

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
- [x] 12.5 个人中心优惠券入口（待集成到个人中心）

---

## 技术债务

- [ ] 统一错误处理
- [ ] 网络请求拦截器优化
- [ ] 图片缓存优化
- [ ] 性能优化（FlatList优化/内存优化）
- [ ] 离线缓存策略
- [ ] 单元测试补充
- [ ] E2E测试补充

---

## 优先级说明

- **P0**: 核心购物链路必须功能，直接影响用户完成购买
- **P1**: 重要增强功能，影响用户体验和留存
- **P2**: 锦上添花功能，提升App完整度

## 预计完成顺序

1. 用户注册/登录页面 (P0-1)
2. 商品搜索页面 (P0-2)
3. 商品列表页 (P0-3)
4. 商品规格/SKU选择 (P0-4)
5. 收货地址管理 (P0-5)
6. 订单列表页 (P0-6)
7. 订单详情页 (P0-7)
8. 物流信息 (P1-8)
9. 确认收货 (P1-9)
10. 退货/退款申请 (P1-10)
11. 收藏功能 (P1-11)
12. 其他次要功能 (P2)
