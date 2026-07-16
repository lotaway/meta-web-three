import http from '@/utils/http'

export interface SalesTrend {
  dates: string[]
  amounts: number[]
  orderCounts: number[]
}

export interface CategoryDistribution {
  categoryName: string
  salesAmount: number
  salesQuantity: number
  growthRate: number
  proportion?: number
}

export interface RegionalComparison {
  region: string
  salesAmount: number
  orderCount: number
  customerCount: number
}

export interface FinancialSummary {
  totalRevenue: number
  totalCost: number
  grossProfit: number
  netProfit: number
  orderCount: number
}

export interface BudgetExecution {
  category: string
  budgetAmount: number
  actualAmount: number
  executionRate: number
}

export interface InventoryTurnover {
  period: string
  turnoverRate: number
  daysInInventory: number
}

export interface AbcAnalysis {
  category: string
  itemCount: number
  inventoryValue: number
  proportion: number
}

export interface SafetyStockAlert {
  productId: string
  productName: string
  warehouseName: string
  quantity: number
  minStock: number
  deficiency: number
}

export interface OeeAnalysis {
  period: string
  availability: number
  performance: number
  quality: number
  oee: number
}

export interface YieldRate {
  period: string
  yieldRate: number
  totalProduced: number
  defective: number
}

export interface PlanAchievement {
  period: string
  achievementRate: number
  plannedQuantity: number
  actualQuantity: number
}

export interface OlapQueryResult {
  columns: string[]
  rows: Record<string, any>[]
  totals?: Record<string, any>
  sql?: string
  queryTimeMs?: number
}

export interface SalesFunnel {
  stage: string
  count: number
  conversionRate: number
}

export interface OlapMetadata {
  domains: string[]
  dimensions: Record<string, string[]>
  metrics: Record<string, string[]>
}

export function getSalesTrend(startDate: string, endDate: string) {
  return http<SalesTrend>({
    url: '/api/v1/analysis/sales/trend',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getCategoryDistribution(startDate: string, endDate: string) {
  return http<CategoryDistribution[]>({
    url: '/api/v1/analysis/sales/category',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getRegionalComparison(startDate: string, endDate: string) {
  return http<OlapQueryResult>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'ORDER',
      dimensions: ['region'],
      metrics: ['total_amount_sum', 'count', 'unique_users'],
      filters: [],
      startTime: startDate ? new Date(startDate).toISOString() : null,
      endTime: endDate ? new Date(endDate).toISOString() : null,
      timeGranularity: null,
      limit: 20,
    },
  })
}

export function getFinancialSummary(startDate: string, endDate: string) {
  return http<FinancialSummary>({
    url: '/api/v1/analysis/dashboard/realtime',
    method: 'get',
  })
}

export function getBudgetExecution(year: number, month: number) {
  return http<BudgetExecution[]>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'ORDER',
      dimensions: ['category'],
      metrics: ['total_amount_sum', 'count'],
      filters: [],
      startTime: new Date(year, month - 1, 1).toISOString(),
      endTime: new Date(year, month, 0).toISOString(),
      timeGranularity: 'MONTH',
      limit: 50,
    },
  })
}

export function getInventoryTurnover() {
  return http<InventoryTurnover[]>({
    url: '/api/v1/analysis/inventory/overview',
    method: 'get',
  })
}

export function getAbcAnalysis() {
  return http<OlapQueryResult>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'INVENTORY',
      dimensions: ['product_name'],
      metrics: ['quantity_sum', 'available_qty_avg'],
      filters: [],
      limit: 50,
    },
  })
}

export function getSafetyStockAlerts() {
  return http<SafetyStockAlert[]>({
    url: '/api/v1/analysis/inventory/low-stock',
    method: 'get',
  })
}

export function getOeeAnalysis() {
  return http<OeeAnalysis[]>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'ORDER',
      dimensions: ['hour'],
      metrics: ['count', 'total_amount_sum'],
      filters: [],
      limit: 30,
    },
  })
}

export function getYieldRate() {
  return http<YieldRate[]>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'ORDER',
      dimensions: ['day'],
      metrics: ['count', 'unique_orders'],
      filters: [],
      limit: 30,
    },
  })
}

export function getPlanAchievement() {
  return http<PlanAchievement[]>({
    url: '/api/olap/query',
    method: 'post',
    data: {
      domain: 'ORDER',
      dimensions: ['day'],
      metrics: ['count', 'total_amount_sum'],
      filters: [],
      limit: 30,
    },
  })
}

export function getOlapMetadata() {
  return http<OlapMetadata>({
    url: '/api/olap/metadata',
    method: 'get',
  })
}

export function getProductionAnalytics(startDate: string, endDate: string) {
  return http<Record<string, any>>({
    url: '/api/analytics/production',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getSalesFunnel() {
  return http<Record<string, any>>({
    url: '/api/olap/sales-funnel',
    method: 'get',
  })
}
