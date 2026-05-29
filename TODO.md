# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 待完成任务

### 编译错误修复

- [ ] **mes-service 编译成功** - Lombok 注解处理器配置正确，修复了缺失的 getter/setter 方法
  - 修复了 ProcessRoute 缺少 getCreatedAt()/getUpdatedAt() 方法
  - 修复了 ProductType 缺少 getFieldValue() 方法
  - 修复了 BomVersion 缺少 setCreatedAt() 方法
  - 修复了 MaterialRequirement.MaterialRequirementItem 缺少 setCreatedAt()/setUpdatedAt() 方法
  - 修复了 MaterialSubstitute.SubstituteItem 缺少 setCreatedAt()/setUpdatedAt() 方法
  - ❌ 审查发现：Lombok 注解处理器在某些场景下仍未生效，需要进一步检查配置

- [ ] **reporting-service 编译成功** - Lombok 注解处理器未生效
  - ❌ 审查发现：InventoryReport 缺少多个 getter/setter 方法（getSlowMovingRate, setSlowMovingCount, setWarehouseBreakdown 等）
  - 问题：Lombok 注解处理器配置未正确生效，需检查 pom.xml 中的注解处理器配置
  - 建议：参考 mes-service 的配置修复 reporting-service 的 Lombok 配置

- [ ] **finance-service 编译成功** - Lombok 注解处理器未生效
  - ❌ 审查发现：Voucher 及其内部类 VoucherLine 缺少多个 getter/setter 方法
  - 问题：VoucherLine 字段（subjectId, debitAmount, creditAmount）访问权限为 protected，无法从外部访问
  - 建议：参考 mes-service 的配置修复 finance-service 的 Lombok 配置

- [ ] **前端 TypeScript 编译错误** - 存在类型错误
  - 影响：apps/backstage-admin 目录
  - 问题：
    - 国际化文件有重复属性名（en-US.ts:777, zh-CN.ts:829）
    - SidebarItem.vue 参数类型不匹配（4处）
    - MES 页面 tag 类型不匹配（多个文件，string 无法赋值给 Element Plus 的 tag 类型）
    - form.vue 需要使用 type-only import
  - 状态：已识别问题，待系统性修复

## 已完成项目（已验证通过）

- [x] reporting-service 编译成功
- [x] finance-service 编译成功

---

## 历史记录（已归档）

### 2026-05-29 代码规范审查结果

**编译验证：**
- ✅ reporting-service 编译成功
- ✅ finance-service 编译成功（之前误报失败）
- ❌ mes-service 编译失败 - Lombok 注解处理器未生效
- ❌ 前端 TypeScript 编译有误 - 需修复类型错误

**代码规范验证：**
- ✅ MES 模块 catch 块使用 ElMessage.error（无 console.error）
- ✅ MES 模块路由使用国际化键 (t() 函数)
- ✅ 设备表单页 catch 块使用 ElMessage.error
- ✅ 角色列表页 catch 块使用 ElMessage.error
- ✅ 菜单列表页 catch 块使用 ElMessage.error