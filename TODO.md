# 检查项

## 前端 @directory:client/ 是否按照 @directory:temp/mall-app-web/ 完成所有功能迁移

## 后端 @directory:server/ 是否按照 @directory:temp/mall/ 完成所有功能迁移

## 检查后台 @directory:temp/mall-admin-web 是否对接了后端 @directory:server/ 所需的管理功能


## 一期：基础设施优化

### 1.1 路径别名 —— 在每个微服务内部添加，对齐参考项目接口路径 ✅

**完成内容**: 通过 Spring `@GetMapping({"/original", "/alias"})` 配置兼容路径，不破坏原有端点。

| 微服务 | 控制器 | 新增别名路径 | 目标方法 |
|--------|--------|-------------|---------|
| product-service | ProductController | `GET /detail/{id}` | `getProduct(id)` |
| product-service | ProductController | `GET /categoryTreeList` | `categoryService.categoryTreeList()` |
| product-service | BrandController | `GET /detail/{id}` | `details(id)` |
| product-service | BrandController | `GET /productList` | `listProductsByBrand(brandId?, keyword?)` |
| order-service | OrderController | `POST /generateOrder` | `create(userId, request)` |
| order-service | OrderController | `GET /detail/{orderId}` | `detail(userId, id)` |
| promotion-service | CouponController | `GET /listByProduct/{productId}` | `listByProduct(productId)` |
| user-service | MemberAddressController | `GET /list` | `list(memberId)` |
| user-service | MemberAddressController | `GET /detail/{id}` | `getById(id)` |

**未执行**: ProductController `POST /product/generateOrder` — 属于 order 域，与 product 无关。

### 1.2 重复实现清理 —— user-service vs user-action-service ✅

**完成内容**:
- 删除 `user-service` 的 3 个内存控制器：`MemberAttentionController`、`ProductCollectionController`、`ReadHistoryController`
- `user-action-service` 新增 `MemberActionController`（路径 `/member/*`），包含 14 个端点，使用 PostgreSQL 持久化
- `user-action-service` 的 `UserActionService` 补充了 7 个缺失的 DB 方法（delete/detail/clear）
- 原有 `/action/*` 路径保持向后兼容
- 访问方式：网关路由 `/user-action-service/member/attention/*`

### 1.3 网关配置调整 ✅

**完成内容**:
- 确认 `UserAuthFilter` 的保护前缀 `/user-service/` 与排除列表路径模式正确（`/{serviceId}/path`）
- 移除 3 条已失效的排除路径（user-service 中已删除的 member 端点）: `/member/readHistory`、`/member/productCollection`、`/member/attention`
- 保留 `/member/address` 排除路径（MemberAddressController 仍在 user-service 中）

---

## 二期：后台管理端接口实施

所有新接口按照微服务规范，添加到对应的微服务中。前端通过 `/{serviceId}/admin-path` 访问。

### 2.1 UMS 模块（user-service）—— 管理员与 RBAC

#### 管理员认证与管理
- [ ] **管理员认证**: 新增 `AdminController`(`/admin`)
  - `POST /admin/login` — 管理员登录
  - `POST /admin/logout` — 管理员登出
  - `GET /admin/info` — 获取管理员信息
  - `POST /admin/register` — 管理员注册
  - `GET /admin/list` — 管理员列表（分页）
  - `POST /admin/update/{id}` — 修改管理员
  - `POST /admin/updateStatus/{id}` — 修改状态
  - `POST /admin/delete/{id}` — 删除管理员
  - `GET /admin/role/{adminId}` — 获取管理员角色
  - `POST /admin/role/update` — 分配角色

#### 角色管理
- [ ] **角色管理**: 新增 `RoleController`(`/role`)
  - CRUD + 分配菜单/资源 + 列表查询

#### 菜单管理
- [ ] **菜单管理**: 新增 `MenuController`(`/menu`)
  - CRUD + 树形列表

#### 资源管理
- [ ] **资源管理**: 新增 `ResourceController`(`/resource`)
  - CRUD + 全量列表

#### 资源分类
- [ ] **资源分类**: 新增 `ResourceCategoryController`(`/resourceCategory`)
  - CRUD + 全量列表

#### 会员等级
- [ ] **会员等级**: 新增 `MemberLevelController`(`/memberLevel`)
  - `GET /memberLevel/list` — 会员等级列表

### 2.2 PMS 模块（product-service）—— 商品管理扩展

- [ ] **批量操作**: `POST /product/update/deleteStatus`, `/update/newStatus`, `/update/recommendStatus`, `/update/publishStatus`, `/update/verifyStatus`
- [ ] **简单列表**: `GET /product/simpleList`
- [ ] **分类扩展**: `GET /productCategory/{id}`, `POST /productCategory/update/navStatus`, `POST /productCategory/update/showStatus`
- [ ] **品牌扩展**: `GET /brand/listAll`, `POST /brand/update/showStatus`, `POST /brand/update/factoryStatus`, `POST /brand/delete/batch`
- [ ] **属性分类**（ProductAttributeCategoryController）: CRUD + listWithAttr
- [ ] **属性扩展**: `GET /productAttribute/attrInfo/{productCategoryId}`
- [ ] **SKU 库存**（SkuStockController）: `GET /sku/{pid}`, `POST /sku/update/{pid}`

### 2.3 OMS 模块（order-service）—— 订单管理扩展

- [ ] **订单管理扩展**: `POST /order/update/delivery`, `/update/close`, `/update/receiverInfo`, `/update/moneyInfo`, `/update/note`
- [ ] **退货原因**（OrderReturnReasonController）: CRUD + 分页列表
- [ ] **公司地址**（CompanyAddressController）: `GET /companyAddress/list`

### 2.4 SMS 模块（promotion-service）—— 营销管理

- [ ] **优惠券管理端**: `POST /coupon/create`, `/update/{id}`, `/delete/{id}`, `GET /coupon/list`, `GET /coupon/{id}`, `GET /couponHistory/list`
- [ ] **广告管理扩展**: `POST /home/advertise/update/status/{id}`
- [ ] **秒杀活动**（FlashPromotionController）: CRUD
- [ ] **秒杀时间段**（FlashPromotionSessionController）: CRUD
- [ ] **秒杀商品关联**（FlashPromotionProductRelationController）: CRUD
- [ ] **首页推荐品牌**（HomeBrandController）: 推荐/取消/列表
- [ ] **首页新品推荐**（HomeNewProductController）: 推荐/取消/列表
- [ ] **首页人气推荐**（HomeRecommendProductController）: 推荐/取消/列表
- [ ] **首页专题推荐**（HomeSubjectController）: 推荐/取消/列表

### 2.5 其他模块

- [ ] **商品专题**（SubjectController）: `GET /subject/listAll`, `/list`
- [ ] **优选专区**（PrefrenceAreaController）: `GET /prefrenceArea/listAll`
- [ ] **OSS/MinIO 文件上传签名**: `GET /aliyun/oss/policy`

---

## 三期：backstage-admin 前端适配

### 3.1 路径与认证适配
- [ ] 更新 `backstage-admin/src/utils/http.ts` 中 `baseURL` 指向网关地址 `http://localhost:10081`
- [ ] 更新所有 API 调用路径为 `/{serviceId}/path` 格式（或保持原路径，通过 nginx 转发到网关）
- [ ] 适配 user-service 的登录流程（`/user-service/sso/login` → 获取 token → 存入 cookie）
- [ ] 确保路由守卫逻辑与新认证流程一致

### 3.2 接口测试与联调
- [ ] 逐个测试 29 个 API 文件对应的接口
- [ ] 确保 CRUD 页面功能正常

---

## 注
- Web3/Passkey/加密货币相关功能不在本次范围
- 所有接口通过 Gateway + Discovery Locator 自动路由访问
- 一期完成后即可开始二期和三期并行开发
