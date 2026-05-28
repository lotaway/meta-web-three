<template>
  <div class="equipment-form-container">
    <el-card>
      <template #header>
        <span>{{ isEdit ? t('mes.equipment.form.titleEdit') : t('mes.equipment.form.titleAdd') }}</span>
      </template>
      
      <el-form :model="form" :rules="rules" ref="formRef" label-width="120px">
        <el-form-item :label="t('mes.equipment.equipmentCode')" prop="equipmentCode">
          <el-input v-model="form.equipmentCode" :placeholder="t('mes.equipment.equipmentCodePlaceholder')" :disabled="isEdit" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.equipmentName')" prop="equipmentName">
          <el-input v-model="form.equipmentName" :placeholder="t('mes.equipment.equipmentNamePlaceholder')" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.equipmentTypeCode')" prop="equipmentTypeCode">
          <el-input v-model="form.equipmentTypeCode" :placeholder="t('mes.equipment.equipmentTypeCode') + t('common.placeholderSuffix')" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.workshopId')" prop="workshopId">
          <el-input v-model="form.workshopId" :placeholder="t('mes.equipment.workshopId') + t('common.placeholderSuffix')" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.workstationId')" prop="workstationId">
          <el-input v-model="form.workstationId" :placeholder="t('mes.equipment.workstationId') + t('common.placeholderSuffix')" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.form.positionX')" prop="positionX">
          <el-input-number v-model="form.positionX" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.form.positionY')" prop="positionY">
          <el-input-number v-model="form.positionY" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.form.positionZ')" prop="positionZ">
          <el-input-number v-model="form.positionZ" :precision="2" :step="0.1" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.ipAddress')" prop="ipAddress">
          <el-input v-model="form.ipAddress" :placeholder="t('mes.equipment.ipAddress') + t('common.placeholderSuffix')" />
        </el-form-item>
        
        <el-form-item :label="t('mes.equipment.macAddress')" prop="macAddress">
          <el-input v-model="form.macAddress" :placeholder="t('mes.equipment.macAddress') + t('common.placeholderSuffix')" />
        </el-form-item>
        
        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="loading">{{ t('mes.equipment.form.submit') }}</el-button>
          <el-button @click="handleBack">{{ t('mes.equipment.form.back') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import type { FormInstance } from 'element-plus'
import type { Equipment, CreateEquipmentRequest, UpdateEquipmentRequest } from '@/apis/equipment'
import { getEquipmentByIdAPI, createEquipmentAPI, updateEquipmentAPI } from '@/apis/equipment'

const { t } = useI18n()
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
  equipmentCode: [{ required: true, message: t('mes.equipment.equipmentCodePlaceholder'), trigger: 'blur' }],
  equipmentName: [{ required: true, message: t('mes.equipment.equipmentNamePlaceholder'), trigger: 'blur' }],
  equipmentTypeCode: [{ required: true, message: t('mes.equipment.equipmentTypeCode') + t('common.requiredSuffix'), trigger: 'blur' }]
}

const loadData = async () => {
  if (isEdit.value && route.query.id) {
    const id = Number(route.query.id)
    try {
      const data = await getEquipmentByIdAPI(id)
      Object.assign(form, data)
    } catch (error) {
      ElMessage.error(t('mes.equipment.form.loadError'))
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
          ElMessage.success(t('mes.equipment.form.updateSuccess'))
        } else {
          await createEquipmentAPI(form as CreateEquipmentRequest)
          ElMessage.success(t('mes.equipment.form.createSuccess'))
        }
        router.push('/mes/equipment')
      } catch (error) {
        ElMessage.error(isEdit.value ? t('mes.equipment.form.updateFailed') : t('mes.equipment.form.createFailed'))
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