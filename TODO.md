# API 迁移待办事项

## 已完成 ✅

### client/api/home.ts
- `fetchMallHomeContent` - 使用 `brandApi.list()` 获取品牌列表
- `fetchProductCategoryList` - 使用 `categoryApi.viewChildren()` 获取分类
- `fetchNewMallProductList` - 使用 `productApi.listProducts({ keyword: 'new' })`
- `fetchHotMallProductList` - 使用 `productApi.listProducts({ keyword: 'hot' })`

### client/api/product.ts
- `fetchProductDetail` - 使用 `productApi.getProduct()`
- `fetchProductList` - 使用 `productApi.listProducts()`

---

## 待补充的 API 接口 (需要网关添加)

### 首页相关 (旧: /home/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 首页内容 | GET /home/content | - | ❌ 缺少 |
| 首页广告轮播 | GET /home/content (advertiseList) | - | ❌ 缺少 |
| 秒杀活动 | GET /home/content (flashPromotion) | - | ❌ 缺少 |
| 新品推荐 | GET /home/newProductList | /v1/products?keyword=new | ⚠️ 临时方案 |
| 热卖推荐 | GET /home/hotProductList | /v1/products?keyword=hot | ⚠️ 临时方案 |
| 推荐商品 | GET /home/recommendProductList | - | ❌ 缺少 |

### 商品相关 (旧: /product/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 商品搜索 | GET /product/search | /v1/products | ✅ 已完成 |
| 分类树 | GET /product/categoryTreeList | - | ❌ 缺少 |

### 品牌相关 (旧: /brand/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 品牌详情 | GET /brand/detail/{id} | GET /v1/brands/{id} | ✅ 已完成 |
| 品牌商品列表 | GET /brand/productList | - | ❌ 缺少 |
| 品牌推荐列表 | GET /brand/recommendList | - | ❌ 缺少 |

### 购物车 (旧: /cart/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 购物车列表 | GET /cart/list | GET /v1/cart/list | ⚠️ 需 X-User-ID header |
| 加入购物车 | POST /cart/add | POST /v1/cart/add | ⚠️ 需 X-User-ID header |
| 删除购物车商品 | POST /cart/delete | DELETE /v1/cart/delete | ⚠️ 需 X-User-ID header |
| 更新商品数量 | GET /cart/update/quantity | PUT /v1/cart/update/quantity | ⚠️ 需 X-User-ID header |
| 清空购物车 | POST /cart/clear | POST /v1/cart/clear | ⚠️ 需 X-User-ID header |

### 订单相关 (旧: /order/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 生成确认订单 | POST /order/generateConfirmOrder | - | ❌ 缺少 |
| 生成订单 | POST /order/generateOrder | POST /order/create | ⚠️ 参数不同 |
| 订单列表 | GET /order/list | GET /order/list | ✅ 已完成 |
| 订单详情 | GET /order/detail/{orderId} | GET /order/{id} | ⚠️ 参数不同 |
| 取消订单 | POST /order/cancelUserOrder | POST /order/cancel/{id} | ⚠️ 参数不同 |
| 确认收货 | POST /order/confirmReceiveOrder | POST /order/confirm-receive/{id} | ⚠️ 参数不同 |
| 删除订单 | POST /order/deleteOrder | - | ❌ 缺少 |
| 支付成功回调 | POST /order/paySuccess | - | ❌ 缺少 |

### 会员相关 (旧: /member/*, /sso/*)
| 功能 | 旧路径 | 新路径 | 状态 |
|------|--------|--------|------|
| 会员登录 | POST /sso/login | - | ❌ 缺少 |
| 会员信息 | GET /sso/info | - | ❌ 缺少 |
| 地址列表 | GET /member/address/list | - | ❌ 缺少 |
| 地址详情 | GET /member/address/{id} | - | ❌ 缺少 |
| 新增地址 | POST /member/address/add | - | ❌ 缺少 |
| 更新地址 | POST /member/address/update/{id} | - | ❌ 缺少 |
| 删除地址 | POST /member/address/delete/{id} | - | ❌ 缺少 |
| 优惠券列表 | GET /member/coupon/list | - | ❌ 缺少 |
| 领取优惠券 | POST /member/coupon/add/{couponId} | - | ❌ 缺少 |
| 商品优惠券 | GET /member/coupon/listByProduct/{productId} | - | ❌ 缺少 |
| 收藏商品 | POST /member/productCollection/add | - | ❌ 缺少 |
| 收藏列表 | GET /member/productCollection/list | - | ❌ 缺少 |
| 取消收藏 | POST /member/productCollection/delete | - | ❌ 缺少 |
| 足迹列表 | GET /member/readHistory/list | - | ❌ 缺少 |
| 品牌关注 | POST /member/brandAttention/add | - | ❌ 缺少 |
| 品牌关注列表 | GET /member/brandAttention/list | - | ❌ 缺少 |

---

## 已知问题

1. **购物车和订单接口** 需要 `X-User-ID` header，需要添加用户认证逻辑
2. **分页参数** 旧 API 使用 `pageSize`/`pageNum`，新 API 使用 `page`/`size`，需要适配
3. **数据模型差异** 新 API 返回的数据结构与旧 API 不同，需要维护映射层

---

## 迁移建议

1. 优先在网关添加缺失的业务接口（首页内容、会员、订单等）
2. 统一分页参数格式
3. 添加用户认证后，统一传递 `X-User-ID`