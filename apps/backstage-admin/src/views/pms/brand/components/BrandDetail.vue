<script lang="ts" setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { createBrandAPI, getBrandAPI, updateBrandAPI } from '@/apis/brand'
import SingleUpload from '@/components/Upload/singleUpload.vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import type { PmsBrand } from '@/types/brand'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const route = useRoute()
const router = useRouter()

// Define properties
const props = defineProps({
  // Whether in edit mode
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Define brand ID
const brandId = computed(() => Number(route.query.id))

// Default brand, used when creating
const defaultBrand: PmsBrand = {
  bigPic: '',
  brandStory: '',
  factoryStatus: 0,
  firstLetter: '',
  logo: '',
  name: '',
  showStatus: 0,
  sort: 0
}
// Define brand object
const brand = ref<PmsBrand>(Object.assign({}, defaultBrand))
// Brand form component ref
const brandFromRef = ref<FormInstance>()
// Brand form validation rules
const rules = reactive<FormRules<typeof brand>>({
  name: [
    { required: true, message: '请输入品牌名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ],
  logo: [
    { required: true, message: '请输入品牌logo', trigger: 'blur' }
  ],
  sort: [
    { type: 'number', message: '排序必须为数字' }
  ],
})

// Load data on mount
onMounted(async () => {
  if (props.isEdit) {
    // In edit mode, fetch brand details via API
    const res = await getBrandAPI(brandId.value)
    brand.value = res.data
  } else {
    // In create mode, use the default brand
    brand.value = Object.assign({}, defaultBrand)
  }
})

const showValidationError = () => {
  ElMessage({
    message: t('pms.brand.validationFailed'),
    type: 'error',
    duration: 1000
  })
}

const confirmBrandSubmit = async () => {
  await ElMessageBox.confirm(t('dialog.confirmSubmit'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
}

const submitBrandUpdate = async () => {
  await updateBrandAPI(brandId.value, brand.value)
  brandFromRef.value!.resetFields()
  ElMessage({
    message: t('pms.brand.modifySuccess'),
    type: 'success',
    duration: 1000
  })
  router.back()
}

const submitBrandCreate = async () => {
  await createBrandAPI(brand.value)
  brandFromRef.value!.resetFields()
  brand.value = Object.assign({}, defaultBrand)
  ElMessage({
    message: t('pms.brand.createSuccess'),
    type: 'success',
    duration: 1000
  })
}

// Handle brand form submission
const handleBrandSubmit = () => {
  brandFromRef.value!.validate(async (valid) => {
    if (!valid) {
      showValidationError()
      return
    }
    await confirmBrandSubmit()
    if (props.isEdit) {
      await submitBrandUpdate()
      return
    }
    await submitBrandCreate()
  })
}

// Handle brand form reset
const handleBrandReset = () => {
  brandFromRef.value!.resetFields()
  brand.value = Object.assign({}, defaultBrand)
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-form :model="brand" :rules="rules" ref="brandFromRef" label-width="150px">
      <el-form-item :label="t('product.brandName')" prop="name">
        <el-input v-model="brand.name"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.brandInitial')">
        <el-input v-model="brand.firstLetter"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.brandLogo')" prop="logo">
        <single-upload v-model="brand.logo"></single-upload>
      </el-form-item>
      <el-form-item :label="t('product.brandBanner')">
        <single-upload v-model="brand.bigPic"></single-upload>
      </el-form-item>
      <el-form-item :label="t('product.brandStory')">
        <el-input :placeholder="t('oms.order.list.notePlaceholder')" type="textarea" v-model="brand.brandStory" :autosize="true"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.sort')" prop="sort">
        <el-input v-model.number="brand.sort"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.isDisplay')">
        <el-radio-group v-model="brand.showStatus">
          <el-radio :label="1">{{ t('role.page.yes') }}</el-radio>
          <el-radio :label="0">{{ t('role.page.no') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.brandManufacturer')">
        <el-radio-group v-model="brand.factoryStatus">
          <el-radio :label="1">{{ t('role.page.yes') }}</el-radio>
          <el-radio :label="0">{{ t('role.page.no') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleBrandSubmit()">{{ t('common.submit') }}</el-button>
        <el-button v-if="!props.isEdit" @click="handleBrandReset()">{{ t('common.reset') }}</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style></style>
