import http from '@/utils/http'

export interface SalesTrendDTO {
  dates: string[]
  amounts: number[]
  orderCounts: number[]
}

export interface SalesStatisticsDTO {
  date: string
  totalAmount: number
  orderCount: number
  avgOrderValue: number
}

export interface CategorySalesDTO {
  categoryId: string
  categoryName: string
  salesAmount: number
  salesQuantity: number
  growthRate: number
}

export interface UserPortraitDTO {
  totalUsers: number
  newUsers: number
  activeUsers: number
  ageDistribution: { age: string; count: number }[]
  genderDistribution: { gender: string; count: number }[]
  regionDistribution: { region: string; count: number }[]
}

export interface UserProfileDTO {
  userId: number
  username: string
  age: number
  gender: string
  region: string
  purchaseFrequency: number
  avgOrderValue: number
  favoriteCategories: string[]
  lastPurchaseDate: string
}

export interface InventoryOverviewDTO {
  totalProducts: number
  totalQuantity: number
  lowStockCount: number
  overstockCount: number
  totalValue: number
}

export interface InventoryAnalysisDTO {
  productId: string
  productName: string
  warehouseId: number
  warehouseName: string
  quantity: number
  minStock: number
  maxStock: number
  turnoverRate: number
  lastUpdateTime: string
}

export function getSalesTrendAPI(startDate: string, endDate: string) {
  return http<SalesTrendDTO>({
    url: '/api/v1/analysis/sales/trend',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getDailySalesAPI(date: string) {
  return http<SalesStatisticsDTO>({
    url: '/api/v1/analysis/sales/daily',
    method: 'get',
    params: { date },
  })
}

export function getCategorySalesAPI(startDate: string, endDate: string) {
  return http<CategorySalesDTO[]>({
    url: '/api/v1/analysis/sales/category',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getUserPortraitAPI(startDate: string, endDate: string) {
  return http<UserPortraitDTO>({
    url: '/api/v1/analysis/user/portrait',
    method: 'get',
    params: { startDate, endDate },
  })
}

export function getUserProfileAPI(userId: number) {
  return http<UserProfileDTO>({
    url: `/api/v1/analysis/user/profile/${userId}`,
    method: 'get',
  })
}

export function getInventoryOverviewAPI() {
  return http<InventoryOverviewDTO>({
    url: '/api/v1/analysis/inventory/overview',
    method: 'get',
  })
}

export function getProductInventoryAPI(productId: string) {
  return http<InventoryAnalysisDTO>({
    url: `/api/v1/analysis/inventory/product/${productId}`,
    method: 'get',
  })
}

export function getLowStockProductsAPI() {
  return http<InventoryAnalysisDTO[]>({
    url: '/api/v1/analysis/inventory/low-stock',
    method: 'get',
  })
}

export function getOverstockProductsAPI() {
  return http<InventoryAnalysisDTO[]>({
    url: '/api/v1/analysis/inventory/overstock',
    method: 'get',
  })
}

export interface HotProductDTO {
  productId: string
  productName: string
  salesCount: number
  salesAmount: number
}

export interface SalesByHourDTO {
  hour: number
  sales: number
  orders: number
}

export interface RealTimeDashboardDTO {
  todaySales: number
  todayOrders: number
  todayVisitors: number
  conversionRate: number
  todayProfit: number
  pendingOrders: number
  lowStockAlerts: number
  pendingPayments: number
  hotProducts: HotProductDTO[]
  salesByHour: SalesByHourDTO[]
  orderStatusDistribution: Record<string, number>
  categorySalesDistribution: Record<string, number>
  weekOverWeekGrowth: number
  monthOverMonthGrowth: number
}

export function getRealTimeDashboardAPI() {
  return http<RealTimeDashboardDTO>({
    url: '/api/v1/analysis/dashboard/realtime',
    method: 'get',
  })
}