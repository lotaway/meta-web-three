<template>
  <div class="equipment-form-container">
    <el-card>
      <template #header>
        <span>{{ isEdit ? '编辑设备' : '新增设备' }}</span>
      </template>
      
      <el-form :model="form" :rules="rules" ref="formRef" label-width="120px">
        <el-form-item label="设备编码" prop="equipmentCode">
          <el-input v-model="form.equipmentCode" placeholder="请输入设备编码" :disabled="isEdit" />
        </el-form-item>
        
        <el-form-item label="设备名称" prop="equipmentName">
          <el-input v-model="form.equipmentName" placeholder="请输入设备名称" />
        </el-form-item>
        
        <el-form-item label="设备类型编码" prop="equipmentTypeCode">
          <el-input v-model="form.equipmentTypeCode" placeholder="请输入设备类型编码" />
        </el-form-item>
        
        <el-form-item label="车间ID" prop="workshopId">
          <el-input v-model="form.workshopId" placeholder="请输入车间ID" />
        </el-form-item>
        
        <el-form-item label="工位ID" prop="workstationId">
          <el-input v-model="form.workstationId" placeholder="请输入工位ID" />
        </el-form-item>
        
        <el-form-item label="X坐标" prop="positionX">
          <el-input-number v-model="form.positionX" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item label="Y坐标" prop="positionY">
          <el-input-number v-model="form.positionY" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item label="Z坐标" prop="positionZ">
          <el-input-number v-model="form.positionZ" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item label="IP地址" prop="ipAddress">
          <el-input v-model="form.ipAddress" placeholder="请输入IP地址" />
        </el-form-item>
        
        <el-form-item label="MAC地址" prop="macAddress">
          <el-input v-model="form.macAddress" placeholder="请输入MAC地址" />
        </el-form-item>
        
        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="loading">提交</el-button>
          <el-button @click="handleBack">返回</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, FormInstance } from 'element-plus'
import type { Equipment, CreateEquipmentRequest, UpdateEquipmentRequest } from '@/apis/equipment'
import { getEquipmentByIdAPI, createEquipmentAPI, updateEquipmentAPI } from '@/apis/equipment'

const router = useRouter()
const route = useRoute()
const formRef = ref<FormInstance>()
const loading = ref(false)

const isEdit = computed(() => !!route.query.id)

const form = reactive<CreateEquipmentRequest & UpdateEquipmentRequest & { id?: number }>({
  equipmentCode: '',
  equipmentName: '',
  equipmentTypeCode: '',
  workshopId: '',
  workstationId: '',
  positionX: 0,
  positionY: 0,
  positionZ: 0,
  ipAddress: '',
  macAddress: ''
})

const rules = {
  equipmentCode: [{ required: true, message: '请输入设备编码', trigger: 'blur' }],
  equipmentName: [{ required: true, message: '请输入设备名称', trigger: 'blur' }],
  equipmentTypeCode: [{ required: true, message: '请输入设备类型编码', trigger: 'blur' }]
}

const loadData = async () => {
  if (isEdit.value && route.query.id) {
    const id = Number(route.query.id)
    try {
      const data = await getEquipmentByIdAPI(id)
      Object.assign(form, data)
    } catch (error) {
      ElMessage.error('加载设备信息失败')
    }
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (valid) {
      loading.value = true
      try {
        if (isEdit.value) {
          const id = Number(route.query.id)
          const updateData: UpdateEquipmentRequest = {
            equipmentName: form.equipmentName,
            equipmentTypeId: form.equipmentTypeId,
            equipmentTypeCode: form.equipmentTypeCode,
            workshopId: form.workshopId,
            workstationId: form.workstationId,
            positionX: form.positionX,
            positionY: form.positionY,
            positionZ: form.positionZ,
            ipAddress: form.ipAddress,
            macAddress: form.macAddress
          }
          await updateEquipmentAPI(id, updateData)
          ElMessage.success('更新成功')
        } else {
          await createEquipmentAPI(form as CreateEquipmentRequest)
          ElMessage.success('创建成功')
        }
        router.push('/mes/equipment')
      } catch (error) {
        ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
      } finally {
        loading.value = false
      }
    }
  })
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.equipment-form-container {
  padding: 20px;
}
</style>