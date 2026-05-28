import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Equipment, EquipmentStatus } from '@/apis/equipment'
import {
  getEquipmentListAPI,
  getEquipmentByIdAPI,
  createEquipmentAPI,
  updateEquipmentAPI,
  deleteEquipmentAPI,
  startTaskAPI,
  completeTaskAPI,
  reportBreakdownAPI,
  repairEquipmentAPI,
  startMaintenanceAPI,
  completeMaintenanceAPI,
  bindWorkstationAPI,
  unbindWorkstationAPI,
  getEquipmentStatusAPI,
  type CreateEquipmentRequest,
  type UpdateEquipmentRequest
} from '@/apis/equipment'

export const useEquipmentStore = defineStore('equipment', () => {
  const equipmentList = ref<Equipment[]>([])
  const currentEquipment = ref<Equipment | null>(null)
  const loading = ref(false)
  const total = ref(0)

  const queryParams = ref({
    workshopId: '',
    status: '' as EquipmentStatus | '',
    equipmentTypeCode: ''
  })

  const fetchEquipmentList = async (params?: {
    workshopId?: string
    status?: EquipmentStatus
    equipmentTypeCode?: string
  }) => {
    loading.value = true
    try {
      const res = await getEquipmentListAPI(params)
      const data = res.data
      equipmentList.value = data || []
      total.value = data?.length || 0
    } catch (error) {
      console.error('获取设备列表失败:', error)
      equipmentList.value = []
      total.value = 0
    } finally {
      loading.value = false
    }
  }

  const fetchEquipmentById = async (id: number) => {
    loading.value = true
    try {
      const res = await getEquipmentByIdAPI(id)
      const data = res.data
      currentEquipment.value = data
      return data
    } catch (error) {
      console.error('获取设备详情失败:', error)
      currentEquipment.value = null
      return null
    } finally {
      loading.value = false
    }
  }

  const createEquipment = async (data: CreateEquipmentRequest) => {
    loading.value = true
    try {
      const result = await createEquipmentAPI(data)
      return result
    } catch (error) {
      console.error('创建设备失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const updateEquipment = async (id: number, data: UpdateEquipmentRequest) => {
    loading.value = true
    try {
      const result = await updateEquipmentAPI(id, data)
      return result
    } catch (error) {
      console.error('更新设备失败:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const deleteEquipment = async (id: number) => {
    loading.value = true
    try {
      await deleteEquipmentAPI(id)
      return true
    } catch (error) {
      console.error('删除设备失败:', error)
      return false
    } finally {
      loading.value = false
    }
  }

  const startTask = async (id: number, taskNo: string) => {
    try {
      const result = await startTaskAPI(id, taskNo)
      return result
    } catch (error) {
      console.error('开始任务失败:', error)
      throw error
    }
  }

  const completeTask = async (id: number) => {
    try {
      const result = await completeTaskAPI(id)
      return result
    } catch (error) {
      console.error('完成任务失败:', error)
      throw error
    }
  }

  const reportBreakdown = async (id: number, reason?: string) => {
    try {
      const result = await reportBreakdownAPI(id, reason)
      return result
    } catch (error) {
      console.error('报告故障失败:', error)
      throw error
    }
  }

  const repair = async (id: number) => {
    try {
      const result = await repairEquipmentAPI(id)
      return result
    } catch (error) {
      console.error('维修设备失败:', error)
      throw error
    }
  }

  const startMaintenance = async (id: number) => {
    try {
      const result = await startMaintenanceAPI(id)
      return result
    } catch (error) {
      console.error('开始保养失败:', error)
      throw error
    }
  }

  const completeMaintenance = async (id: number) => {
    try {
      const result = await completeMaintenanceAPI(id)
      return result
    } catch (error) {
      console.error('完成保养失败:', error)
      throw error
    }
  }

  const bindWorkstation = async (id: number, workstationId: string) => {
    try {
      const result = await bindWorkstationAPI(id, workstationId)
      return result
    } catch (error) {
      console.error('绑定工位失败:', error)
      throw error
    }
  }

  const unbindWorkstation = async (id: number) => {
    try {
      const result = await unbindWorkstationAPI(id)
      return result
    } catch (error) {
      console.error('解绑工位失败:', error)
      throw error
    }
  }

  const getStatus = async (id: number) => {
    try {
      return await getEquipmentStatusAPI(id)
    } catch (error) {
      console.error('获取设备状态失败:', error)
      return null
    }
  }

  const reset = () => {
    equipmentList.value = []
    currentEquipment.value = null
    loading.value = false
    total.value = 0
    queryParams.value = {
      workshopId: '',
      status: '',
      equipmentTypeCode: ''
    }
  }

  return {
    equipmentList,
    currentEquipment,
    loading,
    total,
    queryParams,
    fetchEquipmentList,
    fetchEquipmentById,
    createEquipment,
    updateEquipment,
    deleteEquipment,
    startTask,
    completeTask,
    reportBreakdown,
    repair,
    startMaintenance,
    completeMaintenance,
    bindWorkstation,
    unbindWorkstation,
    getStatus,
    reset
  }
})