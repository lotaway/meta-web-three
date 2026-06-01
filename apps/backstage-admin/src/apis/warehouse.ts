import type { CommonPage } from '@/types/common'
import http from '@/utils/http'

export interface WarehouseDTO {
  id?: number
  warehouseCode: string
  warehouseName: string
  address: string
  province?: string
  city?: string
  district?: string
  contactName?: string
  contactPhone?: string
  managerName?: string
  capacity?: number
  usedCapacity?: number
  status?: string
  createTime?: string
}

export interface InboundOrderDTO {
  orderNo?: string
  warehouseId?: number
  warehouseName?: string
  supplierId?: number
  supplierName?: string
  inboundType?: string
  expectedArrivalTime?: string
  actualArrivalTime?: string
  status?: string
  totalQuantity?: number
  receivedQuantity?: number
  rejectedQuantity?: number
  createTime?: string
  operator?: string
}

export interface OutboundOrderDTO {
  orderNo?: string
  warehouseId?: number
  warehouseName?: string
  orderType?: string
  relatedOrderNo?: string
  expectedDeliveryTime?: string
  actualDeliveryTime?: string
  status?: string
  totalQuantity?: number
  shippedQuantity?: number
  createTime?: string
  operator?: string
}

export interface InventoryDTO {
  id?: number
  warehouseId?: number
  warehouseName?: string
  productId?: number
  productName?: string
  productCode?: string
  skuId?: number
  skuCode?: string
  quantity?: number
  availableQuantity?: number
  lockedQuantity?: number
  damagedQuantity?: number
  lastInboundTime?: string
  lastOutboundTime?: string
}

// Warehouse APIs
export function getWarehouseListAPI(params: { pageNum?: number; pageSize?: number; status?: string }) {
  return http<CommonPage<WarehouseDTO>>({
    url: '/api/warehouse/warehouses',
    method: 'get',
    params: params,
  })
}

export function getWarehouseByIdAPI(id: number) {
  return http<WarehouseDTO>({
    url: '/api/warehouse/warehouses/' + id,
    method: 'get',
  })
}

export function createWarehouseAPI(data: WarehouseDTO) {
  return http({
    url: '/api/warehouse/warehouses',
    method: 'post',
    data: data,
  })
}

export function updateWarehouseAPI(id: number, data: WarehouseDTO) {
  return http({
    url: '/api/warehouse/warehouses/' + id,
    method: 'put',
    data: data,
  })
}

export function deleteWarehouseAPI(id: number) {
  return http({
    url: '/api/warehouse/warehouses/' + id,
    method: 'delete',
  })
}

// Inbound Order APIs
export function getInboundOrderListAPI(params: { pageNum?: number; pageSize?: number; warehouseId?: number; status?: string; startDate?: string; endDate?: string }) {
  return http<CommonPage<InboundOrderDTO>>({
    url: '/api/warehouse/inbound',
    method: 'get',
    params: params,
  })
}

export function getInboundOrderByIdAPI(orderNo: string) {
  return http<InboundOrderDTO>({
    url: '/api/warehouse/inbound/' + orderNo,
    method: 'get',
  })
}

export function createInboundOrderAPI(data: InboundOrderDTO) {
  return http({
    url: '/api/warehouse/inbound',
    method: 'post',
    data: data,
  })
}

export function confirmInboundOrderAPI(orderNo: string) {
  return http({
    url: '/api/warehouse/inbound/' + orderNo + '/confirm',
    method: 'post',
  })
}

export function cancelInboundOrderAPI(orderNo: string) {
  return http({
    url: '/api/warehouse/inbound/' + orderNo + '/cancel',
    method: 'post',
  })
}

// Outbound Order APIs
export function getOutboundOrderListAPI(params: { pageNum?: number; pageSize?: number; warehouseId?: number; status?: string }) {
  return http<CommonPage<OutboundOrderDTO>>({
    url: '/api/warehouse/outbound',
    method: 'get',
    params: params,
  })
}

export function getOutboundOrderByIdAPI(orderNo: string) {
  return http<OutboundOrderDTO>({
    url: '/api/warehouse/outbound/' + orderNo,
    method: 'get',
  })
}

export function createOutboundOrderAPI(data: OutboundOrderDTO) {
  return http({
    url: '/api/warehouse/outbound',
    method: 'post',
    data: data,
  })
}

export function confirmOutboundOrderAPI(orderNo: string) {
  return http({
    url: '/api/warehouse/outbound/' + orderNo + '/confirm',
    method: 'post',
  })
}

export function cancelOutboundOrderAPI(orderNo: string) {
  return http({
    url: '/api/warehouse/outbound/' + orderNo + '/cancel',
    method: 'post',
  })
}

// Inventory APIs
export function getInventoryListAPI(params: { pageNum?: number; pageSize?: number; warehouseId?: number; productName?: string; skuCode?: string }) {
  return http<CommonPage<InventoryDTO>>({
    url: '/api/warehouse/inventory',
    method: 'get',
    params: params,
  })
}

export function getInventoryByIdAPI(id: number) {
  return http<InventoryDTO>({
    url: '/api/warehouse/inventory/' + id,
    method: 'get',
  })
}

export function adjustInventoryAPI(id: number, data: { quantity: number; reason: string }) {
  return http({
    url: '/api/warehouse/inventory/' + id + '/adjust',
    method: 'post',
    data: data,
  })
}

// Quality Inspection APIs
export function getQualityInspectionListAPI(params: { pageNum?: number; pageSize?: number; warehouseId?: number; status?: string }) {
  return http({
    url: '/api/warehouse/quality',
    method: 'get',
    params: params,
  })
}

export function createQualityInspectionAPI(data: any) {
  return http({
    url: '/api/warehouse/quality',
    method: 'post',
    data: data,
  })
}

export function confirmQualityInspectionAPI(id: number, data: { passed: boolean; remark?: string }) {
  return http({
    url: '/api/warehouse/quality/' + id + '/confirm',
    method: 'post',
    data: data,
  })
}