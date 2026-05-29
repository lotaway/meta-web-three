<template>
  <div class="inspection-item-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.inspectionItem.edit') : t('mes.qc.inspectionItem.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="180px">
        <el-form-item :label="t('mes.qc.inspectionItem.itemCode')" prop="itemCode">
          <el-input v-model="formData.itemCode" :disabled="isEdit" :placeholder="t('mes.qc.inspectionItem.itemCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.itemName')" prop="itemName">
          <el-input v-model="formData.itemName" :placeholder="t('mes.qc.inspectionItem.itemNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.inspectionMethod')" prop="inspectionMethod">
          <el-input v-model="formData.inspectionMethod" :placeholder="t('mes.qc.inspectionItem.inspectionMethodPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.equipmentRequired')" prop="equipmentRequired">
          <el-input v-model="formData.equipmentRequired" :placeholder="t('mes.qc.inspectionItem.equipmentRequiredPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.standardValue')" prop="standardValue">
          <el-input v-model="formData.standardValue" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.upperLimit')" prop="upperLimit">
          <el-input-number v-model="formData.upperLimit" :precision="2" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.lowerLimit')" prop="lowerLimit">
          <el-input-number v-model="formData.lowerLimit" :precision="2" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.unit')" prop="unit">
          <el-input v-model="formData.unit" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionItem.status')" prop="status">
          <el-select v-model="formData.status" :placeholder="t('mes.qc.inspectionItem.statusPlaceholder')">
            <el-option :label="t('mes.qc.inspectionItem.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.inspectionItem.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('mes.qc.inspectionItem.save') }}</el-button>
          <el-button @click="handleCancel">{{ t('mes.qc.inspectionItem.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { FormInstance } from 'element-plus'
import type { QcInspectionItem } from '@/apis/qc'
import {
  getInspectionItemByIdAPI,
  createInspectionItemAPI,
  updateInspectionItemAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<QcInspectionItem>({
  itemCode: '',
  itemName: '',
  inspectionMethod: '',
  equipmentRequired: '',
  standardValue: '',
  upperLimit: undefined,
  lowerLimit: undefined,
  unit: '',
  status: 'INACTIVE'
})

const { t } = useI18n()

const rules = {
  itemCode: [
    { required: true, message: t('mes.qc.inspectionItem.itemCodePlaceholder'), trigger: 'blur' }
  ],
  itemName: [
    { required: true, message: t('mes.qc.inspectionItem.itemNamePlaceholder'), trigger: 'blur' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getInspectionItemByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionItem.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateInspectionItemAPI(Number(route.query.id), formData)
        ElMessage.success(t('mes.qc.inspectionItem.updateSuccess'))
      } else {
        await createInspectionItemAPI(formData)
        ElMessage.success(t('mes.qc.inspectionItem.createSuccess'))
      }
      router.push('/mes/inspectionItem')
    } catch (error) {
      ElMessage.error(t('mes.qc.inspectionItem.deleteFailed'))
    } finally {
      submitting.value = false
    }
  })
}

const handleCancel = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.inspection-item-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>