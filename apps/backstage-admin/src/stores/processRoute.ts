import { defineStore } from 'pinia'
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import type { ProcessRoute, ProcessStep } from '@/apis/processRoute'
import {
  getProcessRouteListAPI,
  getProcessRouteByIdAPI,
  getProcessRouteByCodeAPI,
  getProcessRouteByProductAPI,
  createProcessRouteAPI,
  updateProcessRouteAPI,
  deleteProcessRouteAPI,
  activateProcessRouteAPI,
  archiveProcessRouteAPI,
  validateProcessRouteAPI,
  type CreateProcessRouteRequest,
  type UpdateProcessRouteRequest
} from '@/apis/processRoute'

export const useProcessRouteStore = defineStore('processRoute', () => {
  const routeList = ref<ProcessRoute[]>([])
  const currentRoute = ref<ProcessRoute | null>(null)
  const loading = ref(false)
  const total = ref(0)

  const queryParams = ref({
    routeCode: '',
    routeName: '',
    status: '',
    pageNum: 1,
    pageSize: 10
  })

  const fetchRouteList = async (status?: string) => {
    loading.value = true
    try {
      const res = await getProcessRouteListAPI(status)
      const data = res.data
      routeList.value = data || []
      total.value = data?.length || 0
    } catch (error) {
      ElMessage.error('Failed to fetch route list')
      routeList.value = []
      total.value = 0
    } finally {
      loading.value = false
    }
  }

  const fetchRouteById = async (id: number) => {
    loading.value = true
    try {
      const res = await getProcessRouteByIdAPI(id)
      const data = res.data
      currentRoute.value = data
      return data
    } catch (error) {
      ElMessage.error('Failed to fetch route details')
      currentRoute.value = null
      return null
    } finally {
      loading.value = false
    }
  }

  const fetchRouteByCode = async (routeCode: string) => {
    try {
      return await getProcessRouteByCodeAPI(routeCode)
    } catch (error) {
      ElMessage.error('Failed to fetch route by code')
      return null
    }
  }

  const fetchRouteByProduct = async (productCode: string) => {
    try {
      return await getProcessRouteByProductAPI(productCode)
    } catch (error) {
      ElMessage.error('Failed to fetch route by product')
      return []
    }
  }

  const createRoute = async (data: CreateProcessRouteRequest) => {
    loading.value = true
    try {
      const result = await createProcessRouteAPI(data)
      return result
    } catch (error) {
      throw error
    } finally {
      loading.value = false
    }
  }

  const updateRoute = async (id: number, data: UpdateProcessRouteRequest) => {
    loading.value = true
    try {
      const result = await updateProcessRouteAPI(id, data)
      return result
    } catch (error) {
      throw error
    } finally {
      loading.value = false
    }
  }

  const deleteRoute = async (id: number) => {
    loading.value = true
    try {
      await deleteProcessRouteAPI(id)
      return true
    } catch (error) {
      ElMessage.error('Failed to delete route')
      return false
    } finally {
      loading.value = false
    }
  }

  const activateRoute = async (id: number) => {
    try {
      const result = await activateProcessRouteAPI(id)
      return result
    } catch (error) {
      throw error
    }
  }

  const archiveRoute = async (id: number) => {
    try {
      const result = await archiveProcessRouteAPI(id)
      return result
    } catch (error) {
      throw error
    }
  }

  const validateRoute = async (id: number) => {
    try {
      const result = await validateProcessRouteAPI(id)
      return result
    } catch (error) {
      throw error
    }
  }

  const getFirstStep = async (id: number) => {
    try {
      const { getFirstStepAPI } = await import('@/apis/processRoute')
      return await getFirstStepAPI(id)
    } catch (error) {
      ElMessage.error('Failed to fetch first step')
      return null
    }
  }

  const getNextStep = async (id: number, stepNo: number) => {
    try {
      const { getNextStepAPI } = await import('@/apis/processRoute')
      return await getNextStepAPI(id, stepNo)
    } catch (error) {
      ElMessage.error('Failed to fetch next step')
      return null
    }
  }

  const reset = () => {
    routeList.value = []
    currentRoute.value = null
    loading.value = false
    total.value = 0
    queryParams.value = {
      routeCode: '',
      routeName: '',
      status: '',
      pageNum: 1,
      pageSize: 10
    }
  }

  return {
    routeList,
    currentRoute,
    loading,
    total,
    queryParams,
    fetchRouteList,
    fetchRouteById,
    fetchRouteByCode,
    fetchRouteByProduct,
    createRoute,
    updateRoute,
    deleteRoute,
    activateRoute,
    archiveRoute,
    validateRoute,
    getFirstStep,
    getNextStep,
    reset
  }
})