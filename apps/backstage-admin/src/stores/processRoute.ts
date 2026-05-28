import { defineStore } from 'pinia'
import { ref } from 'vue'
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
  // 状态
  const routeList = ref<ProcessRoute[]>([])
  const currentRoute = ref<ProcessRoute | null>(null)
  const loading = ref(false)
  const total = ref(0)

  // 查询参数
  const queryParams = ref({
    routeCode: '',
    routeName: '',
    status: '',
    pageNum: 1,
    pageSize: 10
  })

  // 获取列表
  const fetchRouteList = async (status?: string) => {
    loading.value = true
    try {
      const res = await getProcessRouteListAPI(status)
      const data = res.data
      routeList.value = data || []
      total.value = data?.length || 0
    } catch (error) {
      console.error('获取工艺路线列表失败:', error)
      routeList.value = []
      total.value = 0
    } finally {
      loading.value = false
    }
  }

  // 获取详情
  const fetchRouteById = async (id: number) => {
    loading.value = true
    try {
      const res = await getProcessRouteByIdAPI(id)
      const data = res.data
      currentRoute.value = data
      return data
    } catch (error) {
      console.error('获取工艺路线详情失败:', error)
      currentRoute.value = null
      return null
    } finally {
      loading.value = false
    }
  }

  // 按编码查询
  const fetchRouteByCode = async (routeCode: string) => {
    try {
      return await getProcessRouteByCodeAPI(routeCode)
    } catch (error) {
      console.error('按编码查询失败:', error)
      return null
    }
  }

  // 按产品查询
  const fetchRouteByProduct = async (productCode: string) => {
    try {
      return await getProcessRouteByProductAPI(productCode)
    } catch (error) {
      console.error('按产品查询失败:', error)
      return []
    }
  }

  // 创建
  const createRoute = async (data: CreateProcessRouteRequest) => {
    loading.value = true
    try {
      const result = await createProcessRouteAPI(data)
      return result
    } catch (error) {
      console.error('创建工艺路线失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  // 更新
  const updateRoute = async (id: number, data: UpdateProcessRouteRequest) => {
    loading.value = true
    try {
      const result = await updateProcessRouteAPI(id, data)
      return result
    } catch (error) {
      console.error('更新工艺路线失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  // 删除
  const deleteRoute = async (id: number) => {
    loading.value = true
    try {
      await deleteProcessRouteAPI(id)
      return true
    } catch (error) {
      console.error('删除工艺路线失败:', error)
      return false
    } finally {
      loading.value = false
    }
  }

  // 激活
  const activateRoute = async (id: number) => {
    try {
      const result = await activateProcessRouteAPI(id)
      return result
    } catch (error) {
      console.error('激活工艺路线失败:', error)
      throw error
    }
  }

  // 归档
  const archiveRoute = async (id: number) => {
    try {
      const result = await archiveProcessRouteAPI(id)
      return result
    } catch (error) {
      console.error('归档工艺路线失败:', error)
      throw error
    }
  }

  // 验证
  const validateRoute = async (id: number) => {
    try {
      const result = await validateProcessRouteAPI(id)
      return result
    } catch (error) {
      console.error('验证工艺路线失败:', error)
      throw error
    }
  }

  // 获取首道工序
  const getFirstStep = async (id: number) => {
    try {
      const { getFirstStepAPI } = await import('@/apis/processRoute')
      return await getFirstStepAPI(id)
    } catch (error) {
      console.error('获取首道工序失败:', error)
      return null
    }
  }

  // 获取下一道工序
  const getNextStep = async (id: number, stepNo: number) => {
    try {
      const { getNextStepAPI } = await import('@/apis/processRoute')
      return await getNextStepAPI(id, stepNo)
    } catch (error) {
      console.error('获取下一道工序失败:', error)
      return null
    }
  }

  // 重置状态
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
    // 状态
    routeList,
    currentRoute,
    loading,
    total,
    queryParams,
    // 方法
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