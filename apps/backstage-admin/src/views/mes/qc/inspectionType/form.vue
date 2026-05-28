<template>
  <div class="inspection-type-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? 'Edit Inspection Type' : 'Add Inspection Type' }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="150px">
        <el-form-item label="Inspection Type Code" prop="typeCode">
          <el-input v-model="formData.typeCode" :disabled="isEdit" placeholder="Enter inspection type code" />
        </el-form-item>

        <el-form-item label="Inspection Type Name" prop="typeName">
          <el-input v-model="formData.typeName" placeholder="Enter inspection type name" />
        </el-form-item>

        <el-form-item label="Inspection Category" prop="category">
          <el-select v-model="formData.category" placeholder="请选择检验分类">
            <el-option label="来料检验" value="INCOMING" />
            <el-option label="工序检验" value="PROCESS" />
            <el-option label="最终检验" value="FINAL" />
            <el-option label="出货检验" value="OUTGOING" />
            <el-option label="自定义" value="CUSTOM" />
          </el-select>
        </el-form-item>

        <el-form-item label="描述" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" placeholder="Enter description" />
        </el-form-item>

        <el-form-item label="Applicable Products" prop="applicableProducts">
          <el-input v-model="formData.applicableProducts" placeholder="Enter applicable products, separated by commas" />
        </el-form-item>

        <el-form-item label="Default Sampling Plan" prop="defaultSamplingPlan">
          <el-input v-model="formData.defaultSamplingPlan" placeholder="Example: GB/T 2828.1" />
        </el-form-item>

        <el-form-item label="Default AQL" prop="defaultAql">
          <el-input v-model="formData.defaultAql" placeholder="例如: 0.1, 0.15, 0.25" />
        </el-form-item>

        <el-form-item label="Default Inspection Timeout (Hours)" prop="defaultTimeoutHours">
          <el-input-number v-model="formData.defaultTimeoutHours" :min="1" :max="168" />
        </el-form-item>

        <el-form-item label="Require Quality Certificate" prop="requireCertificate">
          <el-switch v-model="formData.requireCertificate" />
        </el-form-item>

        <el-form-item label="Require Test Report" prop="requireTestReport">
          <el-switch v-model="formData.requireTestReport" />
        </el-form-item>

        <el-form-item label="Sort Order" prop="sortOrder">
          <el-input-number v-model="formData.sortOrder" :min="0" :max="9999" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">Save</el-button>
          <el-button @click="handleCancel">Cancel</el-button>
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
import type { QcInspectionType, InspectionCategory } from '@/apis/qc'
import {
  getInspectionTypeByIdAPI,
  createInspectionTypeAPI,
  updateInspectionTypeAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<QcInspectionType>({
  typeCode: '',
  typeName: '',
  category: 'INCOMING' as InspectionCategory,
  description: '',
  applicableProducts: '',
  defaultSamplingPlan: '',
  defaultAql: '',
  defaultTimeoutHours: 24,
  requireCertificate: false,
  requireTestReport: false,
  sortOrder: 0,
  status: 'INACTIVE'
})

const { t } = useI18n()

const rules = {
  typeCode: [
    { required: true, message: t('mes.inspectionType.form.typeCodeRequired'), trigger: 'blur' }
  ],
  typeName: [
    { required: true, message: t('mes.inspectionType.form.typeNameRequired'), trigger: 'blur' }
  ],
  category: [
    { required: true, message: t('mes.inspectionType.form.categoryRequired'), trigger: 'change' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getInspectionTypeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error('加载数据失败')
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateInspectionTypeAPI(Number(route.query.id), formData)
        ElMessage.success('更新成功')
      } else {
        await createInspectionTypeAPI(formData)
        ElMessage.success('创建成功')
      }
      router.push('/mes/inspectionType')
    } catch (error) {
      ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
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
.inspection-type-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>
