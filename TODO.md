# 检查项

## 前端 @directory:client/ 是否按照 @directory:temp/mall-app-web/ 完成所有功能迁移
- [x] 核心购物流程（首页、分类、购物车、商品详情、订单、地址）
- [x] 基础用户系统（登录、注册、个人中心、消息通知、优惠券）
- [x] 用户行为系统（收藏、足迹、品牌关注）
- [ ] **缺失功能：** 品牌列表与详情页 (`pages/brand/`)
- [ ] **缺失功能：** 钱包/余额管理页面 (`pages/money/`)
- [ ] **缺失功能：** 独立的设置页面 (`pages/set/`)
- [ ] **缺失功能：** 个人详细信息编辑页面 (`pages/userinfo/`)
- [ ] **缺失功能：** 秒杀/限时购相关 UI

## 后端 @directory:server/ 是否按照 @directory:temp/mall/ 完成所有功能迁移
- [x] 核心业务微服务（user, product, cart, order, payment, promotion, user-action）
- [x] 基础管理功能（Admin, Role, Menu, ProductCategory, Brand, OrderReturn/Setting）
- [x] 搜索功能（整合 ES 至 product-service）
- [ ] **缺失逻辑：** 秒杀/限时购业务逻辑 (`SmsFlashPromotion`)
- [ ] **缺失逻辑：** 首页精细化推荐管理 (`HomeBrand`, `HomeNewProduct`, `HomeRecommendProduct`, `HomeRecommendSubject`)
- [ ] **缺失逻辑：** 内容管理系统 (CMS) 功能 (`CmsSubject`, `CmsPrefrenceArea`)

## 检查后台 @directory:temp/mall-admin-web 是否对接了后端 @directory:server/ 所需的管理功能
- [ ] 对接上述缺失的秒杀、首页推荐及 CMS 管理接口
- [ ] 验证现有 Admin/Role/Product/Order 管理接口在微服务架构下的兼容性