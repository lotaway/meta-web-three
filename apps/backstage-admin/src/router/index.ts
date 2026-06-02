import { createRouter, createWebHashHistory } from 'vue-router'
import Layout from '@/views/layout/Layout.vue'
import type { RouteRecordExt } from '@/types/router'

export const constantRouterMap: RouteRecordExt[] = [
  { path: '/404', component: () => import('@/views/normal/404/index.vue'), hidden: true },
  { path: '/login', component: () => import('@/views/normal/login/index.vue'), hidden: true },
  {
    path: '',
    component: Layout,
    redirect: '/home',
    meta: { title: '首页', icon: 'home' },
    children: [
      {
        path: 'home',
        name: 'home',
        component: () => import('@/views/home/index.vue'),
        meta: { title: '首页', icon: 'dashboard' },
      },

    ],
  },
]

export const asyncRouterMap: RouteRecordExt[] = [
  {
    path: '/pms',
    component: Layout,
    redirect: '/pms/product',
    name: 'pms',
    meta: { title: '商品', icon: 'product' },
    children: [
      {
        path: 'product',
        name: 'product',
        component: () => import('@/views/pms/product/index.vue'),
        meta: { title: '商品列表', icon: 'product-list' },
      },
      {
        path: 'addProduct',
        name: 'addProduct',
        component: () => import('@/views/pms/product/add.vue'),
        meta: { title: '添加商品', icon: 'product-add' },
      },
      {
        path: 'updateProduct',
        name: 'updateProduct',
        component: () => import('@/views/pms/product/update.vue'),
        meta: { title: '修改商品', icon: 'product-add' },
        hidden: true,
      },
      {
        path: 'productCate',
        name: 'productCate',
        component: () => import('@/views/pms/productCate/index.vue'),
        meta: { title: '商品分类', icon: 'product-cate' },
      },
      {
        path: 'addProductCate',
        name: 'addProductCate',
        component: () => import('@/views/pms/productCate/add.vue'),
        meta: { title: '添加商品分类' },
        hidden: true,
      },
      {
        path: 'updateProductCate',
        name: 'updateProductCate',
        component: () => import('@/views/pms/productCate/update.vue'),
        meta: { title: '修改商品分类' },
        hidden: true,
      },
      {
        path: 'productAttr',
        name: 'productAttr',
        component: () => import('@/views/pms/productAttr/index.vue'),
        meta: { title: '商品类型', icon: 'product-attr' },
      },
      {
        path: 'productAttrList',
        name: 'productAttrList',
        component: () => import('@/views/pms/productAttr/productAttrList.vue'),
        meta: { title: '商品属性列表' },
        hidden: true,
      },
      {
        path: 'addProductAttr',
        name: 'addProductAttr',
        component: () => import('@/views/pms/productAttr/addProductAttr.vue'),
        meta: { title: '添加商品属性' },
        hidden: true,
      },
      {
        path: 'updateProductAttr',
        name: 'updateProductAttr',
        component: () => import('@/views/pms/productAttr/updateProductAttr.vue'),
        meta: { title: '修改商品属性' },
        hidden: true,
      },
      {
        path: 'brand',
        name: 'brand',
        component: () => import('@/views/pms/brand/index.vue'),
        meta: { title: '品牌管理', icon: 'product-brand' },
      },
      {
        path: 'addBrand',
        name: 'addBrand',
        component: () => import('@/views/pms/brand/add.vue'),
        meta: { title: '添加品牌' },
        hidden: true,
      },
      {
        path: 'updateBrand',
        name: 'updateBrand',
        component: () => import('@/views/pms/brand/update.vue'),
        meta: { title: '编辑品牌' },
        hidden: true,
      },
    ],
  },
  {
    path: '/oms',
    component: Layout,
    redirect: '/oms/order',
    name: 'oms',
    meta: { title: '订单', icon: 'order' },
    children: [
      {
        path: 'order',
        name: 'order',
        component: () => import('@/views/oms/order/index.vue'),
        meta: { title: '订单列表', icon: 'product-list' },
      },
      {
        path: 'orderDetail',
        name: 'orderDetail',
        component: () => import('@/views/oms/order/orderDetail.vue'),
        meta: { title: '订单详情' },
        hidden: true,
      },
      {
        path: 'deliverOrderList',
        name: 'deliverOrderList',
        component: () => import('@/views/oms/order/deliverOrderList.vue'),
        meta: { title: '发货列表' },
        hidden: true,
      },
      {
        path: 'orderSetting',
        name: 'orderSetting',
        component: () => import('@/views/oms/order/setting.vue'),
        meta: { title: '订单设置', icon: 'order-setting' },
      },
      {
        path: 'returnApply',
        name: 'returnApply',
        component: () => import('@/views/oms/apply/index.vue'),
        meta: { title: '退货申请处理', icon: 'order-return' },
      },
      {
        path: 'returnReason',
        name: 'returnReason',
        component: () => import('@/views/oms/apply/reason.vue'),
        meta: { title: '退货原因设置', icon: 'order-return-reason' },
      },
      {
        path: 'returnApplyDetail',
        name: 'returnApplyDetail',
        component: () => import('@/views/oms/apply/applyDetail.vue'),
        meta: { title: '退货申请详情' },
        hidden: true,
      },
      {
        path: 'afterSale',
        name: 'afterSale',
        component: () => import('@/views/oms/afterSale/index.vue'),
        meta: { title: '售后管理', icon: 'after-sale' },
      },
    ],
  },
  // mall-domain management
  {
    path: '/promotion',
    component: Layout,
    redirect: '/promotion/coupon',
    name: 'promotion',
    meta: { title: 'Promotion', icon: 'sms' },
    children: [
      {
        path: 'coupon',
        name: 'coupon',
        component: () => import('@/views/promotion/coupon/index.vue'),
        meta: { title: 'Coupon Management', icon: 'sms-coupon' },
      },
    ],
  },
  {
    path: '/product',
    component: Layout,
    redirect: '/product/index',
    name: 'product',
    meta: { title: 'Product', icon: 'product' },
    children: [
      {
        path: 'index',
        name: 'productManage',
        component: () => import('@/views/product/index.vue'),
        meta: { title: 'Product Management', icon: 'product' },
      },
    ],
  },
  {
    path: '/risk-control',
    component: Layout,
    redirect: '/risk-control/index',
    name: 'risk-control',
    meta: { title: 'Risk Control', icon: 'warning' },
    children: [
      {
        path: 'index',
        name: 'riskControl',
        component: () => import('@/views/risk-control/index.vue'),
        meta: { title: 'Risk Management', icon: 'warning' },
      },
    ],
  },
  {
    path: '/recommendation',
    component: Layout,
    redirect: '/recommendation/index',
    name: 'recommendation',
    meta: { title: 'Recommendation', icon: 'recommend' },
    children: [
      {
        path: 'index',
        name: 'recommendationManage',
        component: () => import('@/views/recommendation/index.vue'),
        meta: { title: 'Recommendation Management', icon: 'recommend' },
      },
    ],
  },
  {
    path: '/sms',
    component: Layout,
    redirect: '/sms/coupon',
    name: 'sms',
    meta: { title: '营销', icon: 'sms' },
    children: [
      {
        path: 'flash',
        name: 'flash',
        component: () => import('@/views/sms/flash/index.vue'),
        meta: { title: '秒杀活动列表', icon: 'sms-flash' },
      },
      {
        path: 'flashSession',
        name: 'flashSession',
        component: () => import('@/views/sms/flash/sessionList.vue'),
        meta: { title: '秒杀时间段列表' },
        hidden: true,
      },
      {
        path: 'selectSession',
        name: 'selectSession',
        component: () => import('@/views/sms/flash/selectSessionList.vue'),
        meta: { title: '秒杀时间段选择' },
        hidden: true,
      },
      {
        path: 'flashProductRelation',
        name: 'flashProductRelation',
        component: () => import('@/views/sms/flash/productRelationList.vue'),
        meta: { title: '秒杀商品列表' },
        hidden: true,
      },
      {
        path: 'coupon',
        name: 'coupon',
        component: () => import('@/views/sms/coupon/index.vue'),
        meta: { title: '优惠券列表', icon: 'sms-coupon' },
      },
      {
        path: 'addCoupon',
        name: 'addCoupon',
        component: () => import('@/views/sms/coupon/add.vue'),
        meta: { title: '添加优惠券' },
        hidden: true,
      },
      {
        path: 'updateCoupon',
        name: 'updateCoupon',
        component: () => import('@/views/sms/coupon/update.vue'),
        meta: { title: '修改优惠券' },
        hidden: true,
      },
      {
        path: 'couponHistory',
        name: 'couponHistory',
        component: () => import('@/views/sms/coupon/history.vue'),
        meta: { title: '优惠券领取详情' },
        hidden: true,
      },
      {
        path: 'brand',
        name: 'homeBrand',
        component: () => import('@/views/sms/brand/index.vue'),
        meta: { title: '品牌推荐', icon: 'product-brand' },
      },
      {
        path: 'new',
        name: 'homeNew',
        component: () => import('@/views/sms/new/index.vue'),
        meta: { title: '新品推荐', icon: 'sms-new' },
      },
      {
        path: 'hot',
        name: 'homeHot',
        component: () => import('@/views/sms/hot/index.vue'),
        meta: { title: '人气推荐', icon: 'sms-hot' },
      },
      {
        path: 'subject',
        name: 'homeSubject',
        component: () => import('@/views/sms/subject/index.vue'),
        meta: { title: '专题推荐', icon: 'sms-subject' },
      },
      {
        path: 'advertise',
        name: 'homeAdvertise',
        component: () => import('@/views/sms/advertise/index.vue'),
        meta: { title: '广告列表', icon: 'sms-ad' },
      },
      {
        path: 'addAdvertise',
        name: 'addHomeAdvertise',
        component: () => import('@/views/sms/advertise/add.vue'),
        meta: { title: '添加广告' },
        hidden: true,
      },
      {
        path: 'updateAdvertise',
        name: 'updateHomeAdvertise',
        component: () => import('@/views/sms/advertise/update.vue'),
        meta: { title: '编辑广告' },
        hidden: true,
      },
      {
        path: 'recommendation',
        name: 'recommendation',
        component: () => import('@/views/sms/recommendation/index.vue'),
        meta: { title: '推荐规则管理', icon: 'sms-recommend' },
      },
    ],
  },
  {
    path: '/ums',
    component: Layout,
    redirect: '/ums/admin',
    name: 'ums',
    meta: { title: '权限', icon: 'ums' },
    children: [
      {
        path: 'admin',
        name: 'admin',
        component: () => import('@/views/ums/admin/index.vue'),
        meta: { title: '用户列表', icon: 'ums-admin' },
      },
      {
        path: 'role',
        name: 'role',
        component: () => import('@/views/ums/role/index.vue'),
        meta: { title: '角色列表', icon: 'ums-role' },
      },
      {
        path: 'allocMenu',
        name: 'allocMenu',
        component: () => import('@/views/ums/role/allocMenu.vue'),
        meta: { title: '分配菜单' },
        hidden: true,
      },
      {
        path: 'allocResource',
        name: 'allocResource',
        component: () => import('@/views/ums/role/allocResource.vue'),
        meta: { title: '分配资源' },
        hidden: true,
      },
      {
        path: 'menu',
        name: 'menu',
        component: () => import('@/views/ums/menu/index.vue'),
        meta: { title: '菜单列表', icon: 'ums-menu' },
      },
      {
        path: 'addMenu',
        name: 'addMenu',
        component: () => import('@/views/ums/menu/add.vue'),
        meta: { title: '添加菜单' },
        hidden: true,
      },
      {
        path: 'updateMenu',
        name: 'updateMenu',
        component: () => import('@/views/ums/menu/update.vue'),
        meta: { title: '修改菜单' },
        hidden: true,
      },
      {
        path: 'resource',
        name: 'resource',
        component: () => import('@/views/ums/resource/index.vue'),
        meta: { title: '资源列表', icon: 'ums-resource' },
      },
      {
        path: 'resourceCategory',
        name: 'resourceCategory',
        component: () => import('@/views/ums/resource/categoryList.vue'),
        meta: { title: '资源分类' },
        hidden: true,
      },
      {
        path: 'memberLevel',
        name: 'memberLevel',
        component: () => import('@/views/ums/memberLevel/index.vue'),
        meta: { title: '会员等级', icon: 'ums-member' },
      },
    ],
  },
  {
    path: '/cs',
    component: Layout,
    redirect: '/cs/dashboard',
    name: 'cs',
    meta: { title: '客服管理', icon: 'service' },
    children: [
      {
        path: 'dashboard',
        name: 'csDashboard',
        component: () => import('@/views/cs/dashboard.vue'),
        meta: { title: '客服工作台', icon: 'cs-dashboard' },
      },
      {
        path: 'agents',
        name: 'csAgents',
        component: () => import('@/views/cs/agents.vue'),
        meta: { title: '客服人员', icon: 'cs-agents' },
      },
      {
        path: 'quick-reply',
        name: 'csQuickReply',
        component: () => import('@/views/cs/quick-reply.vue'),
        meta: { title: '快捷回复', icon: 'cs-quick-reply' },
      },
    ],
  },
  // MES 制造执行系统 - 工艺路线模块
  {
    path: '/mes',
    component: Layout,
    redirect: '/mes/process-route',
    name: 'mes',
    meta: { title: 'mes.title', icon: 'mes' },
    children: [
      {
        path: 'process-route',
        name: 'processRoute',
        component: () => import('@/views/mes/processRoute/index.vue'),
        meta: { title: 'mes.processRoute.title', icon: 'mes-route' },
      },
      {
        path: 'process-route/form',
        name: 'processRouteForm',
        component: () => import('@/views/mes/processRoute/form.vue'),
        meta: { title: 'mes.processRoute.form', hidden: true },
      },
      {
        path: 'process-route/detail',
        name: 'processRouteDetail',
        component: () => import('@/views/mes/processRoute/detail.vue'),
        meta: { title: 'mes.processRoute.detail', hidden: true },
      },
      {
        path: 'equipment',
        name: 'equipment',
        component: () => import('@/views/mes/equipment/index.vue'),
        meta: { title: 'mes.equipment.title', icon: 'mes-equipment' },
      },
      {
        path: 'equipment/form',
        name: 'equipmentForm',
        component: () => import('@/views/mes/equipment/form.vue'),
        meta: { title: 'mes.equipment.form', hidden: true },
      },
      {
        path: 'equipment/detail',
        name: 'equipmentDetail',
        component: () => import('@/views/mes/equipment/detail.vue'),
        meta: { title: 'mes.equipment.detail', hidden: true },
      },
      {
        path: 'equipment/maintenance-plan',
        name: 'equipmentMaintenancePlan',
        component: () => import('@/views/mes/equipment/maintenancePlan.vue'),
        meta: { title: 'mes.equipment.maintenancePlan', icon: 'mes-equipment' },
      },
      {
        path: 'equipment/checklist',
        name: 'equipmentChecklist',
        component: () => import('@/views/mes/equipment/checklist.vue'),
        meta: { title: 'mes.equipment.checklist', icon: 'mes-equipment' },
      },
      {
        path: 'pokayoke',
        name: 'pokayoke',
        component: () => import('@/views/mes/pokayoke/index.vue'),
        meta: { title: 'mes.pokayoke.title', icon: 'mes-pokayoke' },
      },
      {
        path: 'pokayoke/form',
        name: 'pokayokeForm',
        component: () => import('@/views/mes/pokayoke/form.vue'),
        meta: { title: 'mes.pokayoke.form', hidden: true },
      },
      {
        path: 'pokayoke/detail',
        name: 'pokayokeDetail',
        component: () => import('@/views/mes/pokayoke/detail.vue'),
        meta: { title: 'mes.pokayoke.detail', hidden: true },
      },
      // 生产任务模块
      {
        path: 'production-task',
        name: 'productionTask',
        component: () => import('@/views/mes/productionTask/index.vue'),
        meta: { title: 'mes.productionTask.title', icon: 'mes-task' },
      },
      {
        path: 'production-task/form',
        name: 'productionTaskForm',
        component: () => import('@/views/mes/productionTask/form.vue'),
        meta: { title: 'mes.productionTask.form', hidden: true },
      },
      {
        path: 'production-task/detail',
        name: 'productionTaskDetail',
        component: () => import('@/views/mes/productionTask/detail.vue'),
        meta: { title: 'mes.productionTask.detail', hidden: true },
      },
      // 工单管理
      {
        path: 'workOrder',
        name: 'workOrder',
        component: () => import('@/views/mes/workOrder/index.vue'),
        meta: { title: 'mes.workOrder.title', icon: 'mes-workorder' },
      },
      {
        path: 'workOrder/form',
        name: 'workOrderForm',
        component: () => import('@/views/mes/workOrder/form.vue'),
        meta: { title: 'mes.workOrder.form', hidden: true },
      },
      {
        path: 'workOrder/detail',
        name: 'workOrderDetail',
        component: () => import('@/views/mes/workOrder/detail.vue'),
        meta: { title: 'mes.workOrder.detail', hidden: true },
      },
      // QC 质检管理模块
      {
        path: 'inspectionType',
        name: 'inspectionType',
        component: () => import('@/views/mes/qc/inspectionType/index.vue'),
        meta: { title: 'mes.qc.inspectionType.title', icon: 'mes-qc' },
      },
      {
        path: 'inspectionType/form',
        name: 'inspectionTypeForm',
        component: () => import('@/views/mes/qc/inspectionType/form.vue'),
        meta: { title: 'mes.qc.inspectionType.form', hidden: true },
      },
      {
        path: 'inspectionType/detail',
        name: 'inspectionTypeDetail',
        component: () => import('@/views/mes/qc/inspectionType/detail.vue'),
        meta: { title: 'mes.qc.inspectionType.detail', hidden: true },
      },
      {
        path: 'defectCode',
        name: 'defectCode',
        component: () => import('@/views/mes/qc/defectCode/index.vue'),
        meta: { title: 'mes.qc.defectCode.title', icon: 'mes-qc' },
      },
      {
        path: 'defectCode/form',
        name: 'defectCodeForm',
        component: () => import('@/views/mes/qc/defectCode/form.vue'),
        meta: { title: 'mes.qc.defectCode.form', hidden: true },
      },
      {
        path: 'defectCode/detail',
        name: 'defectCodeDetail',
        component: () => import('@/views/mes/qc/defectCode/detail.vue'),
        meta: { title: 'mes.qc.defectCode.detail', hidden: true },
      },
      {
        path: 'nonConformance',
        name: 'nonConformance',
        component: () => import('@/views/mes/qc/nonConformance/index.vue'),
        meta: { title: 'mes.qc.nonConformance.title', icon: 'mes-qc' },
      },
      {
        path: 'nonConformance/form',
        name: 'nonConformanceForm',
        component: () => import('@/views/mes/qc/nonConformance/form.vue'),
        meta: { title: 'mes.qc.nonConformance.form', hidden: true },
      },
      {
        path: 'nonConformance/detail',
        name: 'nonConformanceDetail',
        component: () => import('@/views/mes/qc/nonConformance/detail.vue'),
        meta: { title: 'mes.qc.nonConformance.detail', hidden: true },
      },
      // 流程引擎模块
      {
        path: 'process-template',
        name: 'processTemplate',
        component: () => import('@/views/mes/processTemplate/index.vue'),
        meta: { title: 'mes.processTemplate.title', icon: 'mes-process' },
      },
      {
        path: 'process-template/form',
        name: 'processTemplateForm',
        component: () => import('@/views/mes/processTemplate/form.vue'),
        meta: { title: 'mes.processTemplate.form', hidden: true },
      },
      {
        path: 'process-template/detail',
        name: 'processTemplateDetail',
        component: () => import('@/views/mes/processTemplate/detail.vue'),
        meta: { title: 'mes.processTemplate.detail', hidden: true },
      },
      {
        path: 'process-instance',
        name: 'processInstance',
        component: () => import('@/views/mes/processInstance/index.vue'),
        meta: { title: 'mes.processInstance.title', icon: 'mes-process' },
      },
      {
        path: 'process-instance/detail',
        name: 'processInstanceDetail',
        component: () => import('@/views/mes/processInstance/detail.vue'),
        meta: { title: 'mes.processInstance.detail', hidden: true },
      },
    ],
  },
  // Cash Management Module
  {
    path: '/cash',
    component: Layout,
    redirect: '/cash',
    name: 'cash',
    meta: { title: 'cash.title', icon: 'wallet' },
    children: [
      {
        path: '',
        name: 'cashDashboard',
        component: () => import('@/views/cash/index.vue'),
        meta: { title: 'cash.dashboard', icon: 'cash-dashboard' },
      },
      {
        path: 'plan',
        name: 'cashPlan',
        component: () => import('@/views/cash/plan/index.vue'),
        meta: { title: 'cash.plan.title', icon: 'cash-plan' },
      },
      {
        path: 'account',
        name: 'bankAccount',
        component: () => import('@/views/cash/account/index.vue'),
        meta: { title: 'cash.account.title', icon: 'bank-account' },
      },
      {
        path: 'transfer',
        name: 'cashTransfer',
        component: () => import('@/views/cash/transfer/index.vue'),
        meta: { title: 'cash.transfer.title', icon: 'cash-transfer' },
      },
      {
        path: 'reconciliation',
        name: 'bankReconciliation',
        component: () => import('@/views/cash/reconciliation/index.vue'),
        meta: { title: 'cash.reconciliation.title', icon: 'bank-reconciliation' },
      },
      {
        path: 'forecast',
        name: 'cashFlowForecast',
        component: () => import('@/views/cash/forecast/index.vue'),
        meta: { title: 'cash.forecast.title', icon: 'cash-forecast' },
      },
    ],
  },
  // Inventory Management Module
  {
    path: '/inventory',
    component: Layout,
    redirect: '/inventory/alert',
    name: 'inventory',
    meta: { title: 'inventory.title', icon: 'inventory' },
    children: [
      {
        path: 'alert',
        name: 'inventoryAlert',
        component: () => import('@/views/inventory/inventory-alert/index.vue'),
        meta: { title: 'inventory.alert.title', icon: 'alert' },
      },
    ],
  },

  // Logistics Management Module
  {
    path: '/logistics',
    component: Layout,
    redirect: '/logistics/list',
    name: 'logistics',
    meta: { title: 'logistics.title', icon: 'truck' },
    children: [
      {
        path: 'list',
        name: 'logisticsList',
        component: () => import('@/views/logistics/index.vue'),
        meta: { title: 'logistics.list.title', icon: 'truck' },
      },
    ],
  },
  // Supplier Management Module
  {
    path: '/supplier',
    component: Layout,
    redirect: '/supplier/list',
    name: 'supplier',
    meta: { title: 'supplier.title', icon: 'supplier' },
    children: [
      {
        path: 'list',
        name: 'supplierList',
        component: () => import('@/views/supplier/index.vue'),
        meta: { title: 'supplier.list.title', icon: 'supplier' },
      },
    ],
  },
  // Review Management Module
  {
    path: '/review',
    component: Layout,
    redirect: '/review/list',
    name: 'review',
    meta: { title: 'review.title', icon: 'comment' },
    children: [
      {
        path: 'list',
        name: 'reviewList',
        component: () => import('@/views/review/index.vue'),
        meta: { title: 'review.list.title', icon: 'comment' },
      },
    ],
  },
  // HR Management Module
  {
    path: '/hrm',
    component: Layout,
    redirect: '/hrm/employee',
    name: 'hrm',
    meta: { title: 'hrm.title', icon: 'user' },
    children: [
      {
        path: 'employee',
        name: 'hrmEmployee',
        component: () => import('@/views/hrm/index.vue'),
        meta: { title: 'hrm.employee.title', icon: 'user' },
      },
    ],
  },
  // AI Service Management Module
  {
    path: '/ai',
    component: Layout,
    redirect: '/ai/routing',
    name: 'ai',
    meta: { title: 'ai.title', icon: 'ai' },
    children: [
      {
        path: 'routing',
        name: 'aiRouting',
        component: () => import('@/views/ai/route-optimizer/index.vue'),
        meta: { title: 'ai.routing.title', icon: 'route' },
      },
      {
        path: 'forecasting',
        name: 'aiForecasting',
        component: () => import('@/views/ai/forecasting/index.vue'),
        meta: { title: 'ai.forecasting.title', icon: 'forecast' },
      },
    ],
  },
  // Blockchain Service Management Module
  {
    path: '/blockchain',
    component: Layout,
    redirect: '/blockchain/traceability',
    name: 'blockchain',
    meta: { title: 'blockchain.title', icon: 'blockchain' },
    children: [
      {
        path: 'traceability',
        name: 'blockchainTraceability',
        component: () => import('@/views/blockchain/traceability/index.vue'),
        meta: { title: 'blockchain.traceability.title', icon: 'trace' },
      },
      {
        path: 'wallet',
        name: 'blockchainWallet',
        component: () => import('@/views/blockchain/wallet/index.vue'),
        meta: { title: 'blockchain.wallet.title', icon: 'wallet' },
      },
    ],
  },
]

// createWebHistory（History 模式）地址格式（需要服务器配置）：http://domain.com/admin/home
// createWebHashHistory（Hash 模式）地址格式（会多带一个#号）：http://domain.com/admin/#/home
const router = createRouter({
  history: createWebHashHistory(),
  routes: constantRouterMap,
})

export default router
