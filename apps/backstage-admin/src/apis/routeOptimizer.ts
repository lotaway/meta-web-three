import http from '@/utils/http'

export interface RoutePlan {
  id: number
  planName: string
  vehicleCode: string
  optimizationType: 'DISTANCE' | 'TIME' | 'COST' | 'BALANCED'
  status: 'DRAFT' | 'PENDING' | 'OPTIMIZING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED'
  totalDistance?: number
  totalDuration?: number
  totalCost?: number
  createdAt: string
  updatedAt?: string
}

export interface RoutePoint {
  id?: number
  pointName: string
  pointType: 'PICKUP' | 'DELIVERY' | 'WAREHOUSE' | 'INTERMEDIATE'
  address: string
  latitude?: number
  longitude?: number
  sequence: number
  estimatedArrival?: string
  actualArrival?: string
  status: 'PENDING' | 'VISITED' | 'SKIPPED'
}

export interface Vehicle {
  id: number
  vehicleCode: string
  vehicleNumber: string
  vehicleType: string
  maxLoadCapacity: number
  driverName: string
  driverPhone: string
  status: 'IDLE' | 'ASSIGNED' | 'IN_TRANSIT' | 'MAINTENANCE'
  currentLatitude?: number
  currentLongitude?: number
  lastUpdateTime?: string
}

export interface RouteQueryParam {
  pageNum?: number
  pageSize?: number
  status?: string
  vehicleCode?: string
}

// Route Plan APIs
export function getRoutePlansAPI(params?: RouteQueryParam) {
  return http<RoutePlan[]>({ url: '/api/v1/route-optimizer/routes', method: 'get', params })
}

export function getRoutePlanByIdAPI(id: number) {
  return http<RoutePlan>({ url: `/api/v1/route-optimizer/routes/${id}`, method: 'get' })
}

export function getRoutesByStatusAPI(status: string) {
  return http<RoutePlan[]>({ url: `/api/v1/route-optimizer/routes/status/${status}`, method: 'get' })
}

export function createRoutePlanAPI(data: { planName: string; vehicleCode: string; optimizationType: string }) {
  return http<RoutePlan>({ url: '/api/v1/route-optimizer/routes', method: 'post', data })
}

export function optimizeRouteAPI(id: number) {
  return http<RoutePlan>({ url: `/api/v1/route-optimizer/routes/${id}/optimize`, method: 'post' })
}

export function assignVehicleAPI(id: number, vehicleCode: string) {
  return http<RoutePlan>({ url: `/api/v1/route-optimizer/routes/${id}/assign`, method: 'post', data: { vehicleCode } })
}

export function startRouteAPI(id: number) {
  return http<RoutePlan>({ url: `/api/v1/route-optimizer/routes/${id}/start`, method: 'post' })
}

export function completeRouteAPI(id: number) {
  return http<RoutePlan>({ url: `/api/v1/route-optimizer/routes/${id}/complete`, method: 'post' })
}

// Vehicle APIs
export function getVehiclesAPI() {
  return http<Vehicle[]>({ url: '/api/v1/route-optimizer/vehicles', method: 'get' })
}

export function getVehicleByIdAPI(id: number) {
  return http<Vehicle>({ url: `/api/v1/route-optimizer/vehicles/${id}`, method: 'get' })
}

export function getAvailableVehiclesAPI() {
  return http<Vehicle[]>({ url: '/api/v1/route-optimizer/vehicles/available', method: 'get' })
}

export function getVehiclesByStatusAPI(status: string) {
  return http<Vehicle[]>({ url: `/api/v1/route-optimizer/vehicles/status/${status}`, method: 'get' })
}

export function createVehicleAPI(data: {
  vehicleCode: string
  vehicleNumber: string
  vehicleType: string
  maxLoadCapacity: number
  driverName: string
  driverPhone: string
}) {
  return http<Vehicle>({ url: '/api/v1/route-optimizer/vehicles', method: 'post', data })
}

export function updateVehicleLocationAPI(vehicleCode: string, latitude: number, longitude: number) {
  return http<{ success: boolean }>({
    url: `/api/v1/route-optimizer/vehicles/${vehicleCode}/location`,
    method: 'put',
    data: { latitude, longitude },
  })
}

// Dashboard APIs
export function getPendingRoutesAPI() {
  return http<RoutePlan[]>({ url: '/api/v1/route-optimizer/dashboard/pending-routes', method: 'get' })
}

export function getInProgressRoutesAPI() {
  return http<RoutePlan[]>({ url: '/api/v1/route-optimizer/dashboard/in-progress-routes', method: 'get' })
}

export function getCompletedRoutesAPI() {
  return http<RoutePlan[]>({ url: '/api/v1/route-optimizer/dashboard/completed-routes', method: 'get' })
}