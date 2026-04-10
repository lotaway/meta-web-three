# 待办事项

## client (Expo RN)

### 首页
- 接入 `/v1/home/content` 聚合接口，替换当前 brand + keyword 拼装逻辑
- 接入 `/v1/promotion/advertises` 轮播接口
- `CATE_ITEMS` 从接口获取，调用 `fetchProductCategoryList`

### 分类
- 子分类点击改为进入分类商品列表页

### 商品详情
- `detailMobileHtml` 从接口获取，替换固定空值
- 评价数从接口获取，替换写死数值
- 接入加购、立即购买、收藏接口

### 购物车
- 从 `cartApi` 获取数据，替换本地假数据

### 我的
- 接入用户信息、订单列表、地址、收藏、足迹接口

## 全链路
- 统一环境变量名为 `EXPO_PUBLIC_*`
- 统一鉴权方式（`X-User-ID`）
