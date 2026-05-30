import http from '@/utils/http'

// Fixed Asset Types
export interface FixedAsset {
  id: number
  assetCode: string
  assetName: string
  assetCategory: string
  specification: string
  model: string
  serialNumber: string
  supplierId: number
  supplierName: string
  manufacturer: string
  purchaseDate: string
  originalValue: number
  residualValue: number
  usefulLife: number
  depreciationMethod: string
  departmentId: number
  departmentName: string
  location: string
  custodian: string
  status: string
  usageStatus: string
  remark: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  accumulatedDepreciation: number
  netValue: number
  monthlyDepreciation: number
  annualDepreciationRate: number
}

export interface FixedAssetCreateParams {
  assetCode: string
  assetName: string
  assetCategory: string
  specification?: string
  model?: string
  serialNumber?: string
  supplierId?: number
  supplierName?: string
  manufacturer?: string
  purchaseDate: string
  originalValue: number
  residualValue?: number
  usefulLife: number
  depreciationMethod: string
  departmentId?: number
  departmentName?: string
  location?: string
  custodian?: string
  remark?: string
  createdBy?: number
  creatorName?: string
}

export interface FixedAssetUpdateParams extends FixedAssetCreateParams {
  id: number
}

export interface AssetTransferParams {
  newDepartmentId: number
  newDepartmentName: string
  newLocation: string
  newCustodian: string
}

export interface AssetStatistics {
  totalCount: number
  totalOriginalValue: number
  totalNetValue: number
  totalAccumulatedDepreciation: number
  categoryDistribution: { category: string; count: number; value: number }[]
  departmentDistribution: { departmentId: number; departmentName: string; count: number }[]
  statusDistribution: { status: string; count: number }[]
}

// Depreciation Types
export interface FixedAssetDepreciation {
  id: number
  assetId: number
  assetCode: string
  assetName: string
  depreciationPeriod: string
  depreciationMethod: string
  originalValue: number
  residualValue: number
  usefulLife: number
  depreciationAmount: number
  accumulatedDepreciation: number
  netBookValue: number
  depreciationDate: string
  status: string
  createdAt: string
  updatedAt: string
}

export interface DepreciationGenerateParams {
  departmentId?: number
  depreciationMethod: string
  depreciationPeriod: string
}

export interface DepreciationStatistics {
  totalDepreciationAmount: number
  totalAccumulatedDepreciation: number
  periodDistribution: { period: string; amount: number }[]
}

// Inventory Types
export interface FixedAssetInventory {
  id: number
  inventoryCode: string
  inventoryName: string
  inventoryDate: string
  departmentId: number
  departmentName: string
  inventoryPerson: string
  assetId: number
  assetCode: string
  bookLocation: string
  actualLocation: string
  inventoryResult: string
  discrepancyReason: string
  handleMethod: string
  handleResult: string
  status: string
  remark: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
}

export interface AssetInventoryCreateParams {
  inventoryCode: string
  inventoryName: string
  inventoryDate: string
  departmentId: number
  departmentName: string
  inventoryPerson: string
  assetId: number
  assetCode: string
  bookLocation: string
  actualLocation: string
  inventoryResult?: string
  discrepancyReason?: string
  handleMethod?: string
  remark?: string
  createdBy?: number
  creatorName?: string
}

export interface ConfirmInventoryParams {
  handleResult: string
}

export interface InventoryStatistics {
  totalCount: number
  pendingCount: number
  completedCount: number
  discrepancyCount: number
  resultDistribution: { result: string; count: number }[]
}

// Disposal Types
export interface FixedAssetDisposal {
  id: number
  disposalCode: string
  disposalType: string
  assetId: number
  assetCode: string
  assetName: string
  originalValue: number
  netValue: number
  accumulatedDepreciation: number
  disposalAmount: number
  disposalDate: string
  disposalReason: string
  disposalMethod: string
  acquirerName: string
  acquirerContact: string
  gainLoss: number
  status: string
  approvalStatus: string
  approvalComment: string
  approverId: number
  approverName: string
  approvalDate: string
  remark: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
}

export interface AssetDisposalCreateParams {
  disposalCode: string
  disposalType: string
  assetId: number
  disposalAmount: number
  disposalDate: string
  disposalReason: string
  disposalMethod: string
  acquirerName?: string
  acquirerContact?: string
  remark?: string
  createdBy?: number
  creatorName?: string
}

export interface ApproveDisposalParams {
  approverId: number
  approverName: string
  comment?: string
}

// API Functions
export function listAssets(params?: { departmentId?: number; status?: string; category?: string }) {
  return http<FixedAsset[]>({
    url: '/api/fixed-asset/list',
    method: 'get',
    params
  })
}

export function getAsset(id: number) {
  return http<FixedAsset>({
    url: `/api/fixed-asset/${id}`,
    method: 'get'
  })
}

export function getAssetByCode(code: string) {
  return http<FixedAsset>({
    url: `/api/fixed-asset/code/${code}`,
    method: 'get'
  })
}

export function createAsset(data: FixedAssetCreateParams) {
  return http<{ id: number; success: boolean }>({
    url: '/api/fixed-asset',
    method: 'post',
    data
  })
}

export function updateAsset(id: number, data: FixedAssetUpdateParams) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/${id}`,
    method: 'put',
    data
  })
}

export function deleteAsset(id: number) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/${id}`,
    method: 'delete'
  })
}

export function transferAsset(id: number, data: AssetTransferParams) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/transfer/${id}`,
    method: 'post',
    data
  })
}

export function getAssetStatistics() {
  return http<AssetStatistics>({
    url: '/api/fixed-asset/statistics',
    method: 'get'
  })
}

export function getAssetDepreciation(assetId: number) {
  return http<FixedAssetDepreciation[]>({
    url: `/api/fixed-asset/${assetId}/depreciation`,
    method: 'get'
  })
}

export function generateDepreciation(data: DepreciationGenerateParams) {
  return http<{ success: boolean }>({
    url: '/api/fixed-asset/depreciation/generate',
    method: 'post',
    data
  })
}

export function listDepreciationByPeriod(period: string) {
  return http<FixedAssetDepreciation[]>({
    url: '/api/fixed-asset/depreciation/list',
    method: 'get',
    params: { period }
  })
}

export function getDepreciationStatistics(period: string) {
  return http<DepreciationStatistics>({
    url: '/api/fixed-asset/depreciation/statistics',
    method: 'get',
    params: { period }
  })
}

export function createInventory(data: AssetInventoryCreateParams) {
  return http<{ id: number; success: boolean }>({
    url: '/api/fixed-asset/inventory',
    method: 'post',
    data
  })
}

export function confirmInventory(id: number, data: ConfirmInventoryParams) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/inventory/${id}/confirm`,
    method: 'post',
    data
  })
}

export function listInventory(params?: { status?: string }) {
  return http<FixedAssetInventory[]>({
    url: '/api/fixed-asset/inventory/list',
    method: 'get',
    params
  })
}

export function getInventoryStatistics() {
  return http<InventoryStatistics>({
    url: '/api/fixed-asset/inventory/statistics',
    method: 'get'
  })
}

export function createDisposal(data: AssetDisposalCreateParams) {
  return http<{ id: number; success: boolean }>({
    url: '/api/fixed-asset/disposal',
    method: 'post',
    data
  })
}

export function approveDisposal(id: number, data: ApproveDisposalParams) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/disposal/${id}/approve`,
    method: 'post',
    data
  })
}

export function rejectDisposal(id: number, data: ApproveDisposalParams) {
  return http<{ success: boolean }>({
    url: `/api/fixed-asset/disposal/${id}/reject`,
    method: 'post',
    data
  })
}

export function listDisposal(params?: { status?: string }) {
  return http<FixedAssetDisposal[]>({
    url: '/api/fixed-asset/disposal/list',
    method: 'get',
    params
  })
}