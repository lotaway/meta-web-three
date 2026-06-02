import http from '@/utils/http'

// Warehouse DTO
export interface WarehouseDTO {
  id?: number
  warehouseCode: string
  warehouseName: string
  warehouseType?: string
  province?: string
  city?: string
  district?: string
  address?: string
  contact?: string
  phone?: string
  totalCapacity?: number
  usedCapacity?: number
  availableCapacity?: number
  status?: string
  createdAt?: string
  updatedAt?: string
  // Extended fields for admin UI
  contactName?: string
  contactPhone?: string
  managerName?: string
  capacity?: number
}

// Page response wrapper
interface PageResponse<T> {
  list: T[]
  total: number
}

// Inbound Order DTO
export interface InboundOrderDTO {
  id?: number
  orderNo?: string
  warehouseId?: number
  warehouseName?: string
  inboundType?: string
  supplierId?: number
  supplierName?: string
  status?: string
  expectedArrivalDate?: string
  actualArrivalDate?: string
  totalQuantity?: number
  receivedQuantity?: number
  remark?: string
  createdBy?: string
  createdAt?: string
  updatedAt?: string
  items?: InboundOrderItemDTO[]
}

export interface InboundOrderItemDTO {
  id?: number
  orderNo?: string
  skuCode?: string
  skuName?: string
  quantity?: number
  receivedQuantity?: number
  locationId?: number
  locationCode?: string
  status?: string
}

// Outbound Order DTO
export interface OutboundOrderDTO {
  id?: number
  orderNo?: string
  warehouseId?: number
  warehouseName?: string
  outboundType?: string
  sourceOrderNo?: string
  customerId?: number
  customerName?: string
  status?: string
  expectedDeliveryDate?: string
  actualDeliveryDate?: string
  totalQuantity?: number
  pickedQuantity?: number
  shippedQuantity?: number
  remark?: string
  createdBy?: string
  createdAt?: string
  updatedAt?: string
  items?: OutboundOrderItemDTO[]
}

export interface OutboundOrderItemDTO {
  id?: number
  orderNo?: string
  skuCode?: string
  skuName?: string
  quantity?: number
  pickedQuantity?: number
  locationId?: number
  locationCode?: string
  status?: string
}

// Inventory DTO
export interface InventoryDTO {
  id?: number
  skuCode?: string
  skuName?: string
  warehouseId?: number
  warehouseName?: string
  locationId?: number
  locationCode?: string
  quantity?: number
  availableQuantity?: number
  frozenQuantity?: number
  unit?: string
  status?: string
  lastInboundDate?: string
  lastOutboundDate?: string
  createdAt?: string
  updatedAt?: string
}

// Warehouse APIs
export function getWarehouseListAPI(params?: { pageNum?: number; pageSize?: number; status?: string }) {
  return http<PageResponse<WarehouseDTO>>({
    url: '/api/warehouse/warehouses',
    method: 'get',
    params
  })
}

export function getWarehouseByIdAPI(id: number) {
  return http<WarehouseDTO>({
    url: `/api/warehouse/warehouses/${id}`,
    method: 'get'
  })
}

export function createWarehouseAPI(data: Partial<WarehouseDTO>) {
  return http<number>({
    url: '/api/warehouse/warehouses',
    method: 'post',
    data
  })
}

export function updateWarehouseAPI(id: number, data: Partial<WarehouseDTO>) {
  return http<void>({
    url: `/api/warehouse/warehouses/${id}`,
    method: 'put',
    data
  })
}

export function deleteWarehouseAPI(id: number) {
  return http<void>({
    url: `/api/warehouse/warehouses/${id}`,
    method: 'delete'
  })
}

// Inbound Order APIs
export function getInboundOrderListAPI(params?: { pageNum?: number; pageSize?: number; warehouseId?: number; status?: string }) {
  return http<PageResponse<InboundOrderDTO>>({
    url: '/api/warehouse/inbound',
    method: 'get',
    params
  })
}

export function getInboundOrderByNoAPI(orderNo: string) {
  return http<InboundOrderDTO>({
    url: `/api/warehouse/inbound/${orderNo}`,
    method: 'get'
  })
}

export function createInboundOrderAPI(data: Partial<InboundOrderDTO>) {
  return http<string>({
    url: '/api/warehouse/inbound',
    method: 'post',
    data
  })
}

export function confirmInboundOrderAPI(orderNo: string) {
  return http<void>({
    url: `/api/warehouse/inbound/${orderNo}/confirm`,
    method: 'post'
  })
}

export function completeInboundOrderAPI(orderNo: string, data: Partial<InboundOrderDTO>) {
  return http<void>({
    url: `/api/warehouse/inbound/${orderNo}/complete`,
    method: 'post',
    data
  })
}

// Inventory APIs
export function getInventoryListAPI(params?: { pageNum?: number; pageSize?: number; skuCode?: string; warehouseId?: number }) {
  return http<PageResponse<InventoryDTO>>({
    url: '/api/inventory',
    method: 'get',
    params
  })
}

export function getInventoryBySkuAPI(skuCode: string, warehouseId: number) {
  return http<InventoryDTO>({
    url: '/api/inventory',
    method: 'get',
    params: { skuCode, warehouseId }
  })
}

export function getInventoryBySkuCodeAPI(skuCode: string) {
  return http<InventoryDTO[]>({
    url: `/api/inventory/sku/${skuCode}`,
    method: 'get'
  })
}

export function increaseInventoryAPI(skuCode: string, warehouseId: number, quantity: number, remark?: string) {
  return http<void>({
    url: '/api/inventory/increase',
    method: 'post',
    params: { skuCode, warehouseId, quantity, remark }
  })
}

export function decreaseInventoryAPI(skuCode: string, warehouseId: number, quantity: number, remark?: string) {
  return http<void>({
    url: '/api/inventory/decrease',
    method: 'post',
    params: { skuCode, warehouseId, quantity, remark }
  })
}

// Outbound Order APIs (if available in future)
export function getOutboundOrderListAPI(params?: { pageNum?: number; pageSize?: number; warehouseId?: number; status?: string }) {
  return http<PageResponse<OutboundOrderDTO>>({
    url: '/api/warehouse/outbound',
    method: 'get',
    params
  })
}