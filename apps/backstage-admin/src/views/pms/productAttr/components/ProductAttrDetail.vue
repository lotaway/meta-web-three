<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { getProductAttributeCategoryListAPI } from '@/apis/productAttrCate'
import { productAttributeCreateAPI, getProductAttributeByIdAPI, productAttributeUpdateAPI } from '@/apis/productAttr'
import type { PmsProductAttribute, PmsProductAttributeCategory } from '@/types/productAttr'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const route = useRoute()
const router = useRouter()

// Define properties
const props = defineProps({
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Default product attributes
const defaultProductAttr: PmsProductAttribute = {
  filterType: 0,
  handAddStatus: 0,
  inputList: '',
  inputType: 0,
  name: '',
  productAttributeCategoryId: 0,
  relatedStatus: 0,
  searchType: 0,
  selectType: 0,
  sort: 0,
  type: 0
}
// Current product attribute
const productAttr = ref(Object.assign({}, defaultProductAttr))
// Product attribute category list
const productAttrCateList = ref<PmsProductAttributeCategory[]>([])
// Format attribute value input method
const inputListFormat = computed({
  get() {
    // Convert comma-separated string to newline-separated for display
    return productAttr.value.inputList ? productAttr.value.inputList.replace(/,/g, '\n') : ''
  },
  set(value) {
    // Convert newline-separated string to comma-separated for storage
    productAttr.value.inputList = value ? value.replace(/\n/g, ',') : ''
  }
})
// Get category list
const getCateList = async () => {
  const listQuery = { pageNum: 1, pageSize: 100 }
  const res = await getProductAttributeCategoryListAPI(listQuery)
  productAttrCateList.value = res.data.list
}

// Load data on mount
onMounted(async () => {
  if (props.isEdit) {
    const res = await getProductAttributeByIdAPI(Number(route.query.id))
    productAttr.value = res.data
  } else {
    resetProductAttr()
  }
  getCateList()
})

// Product attribute form component ref
const productAttrFrom = ref<FormInstance>()
// Product attribute form validation rules
const rules = reactive<FormRules<typeof productAttr>>({
  name: [
    { required: true, message: '请输入属性名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ]
})

// Reset product attributes
const resetProductAttr = () => {
  productAttr.value = Object.assign({}, defaultProductAttr)
  productAttr.value.productAttributeCategoryId = Number(route.query.cid)
  productAttr.value.type = Number(route.query.type)
}

// Handle form submission
const confirmAndSubmitAttribute = async () => {
  await ElMessageBox.confirm(t('dialog.confirmSubmit'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  if (props.isEdit) {
    await productAttributeUpdateAPI(Number(route.query.id), productAttr.value)
    ElMessage.success(t('message.modifySuccess'))
    router.back()
    return
  }
  await productAttributeCreateAPI(productAttr.value)
  ElMessage.success(t('message.operationSuccess'))
  resetForm()
}

const onSubmit = async () => {
  if (!productAttrFrom.value) return
  const valid = await productAttrFrom.value.validate().catch(() => false)
  if (!valid) {
    ElMessage.error(t('productAttr.verifyFailed'))
    return
  }
  await confirmAndSubmitAttribute()
}

// Reset form
const resetForm = () => {
  if (productAttrFrom.value) {
    productAttrFrom.value.resetFields()
    resetProductAttr()
  }
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-form :model="productAttr" :rules="rules" ref="productAttrFrom" label-width="150px">
      <el-form-item :label="t('product.attrName')" prop="name">
        <el-input v-model="productAttr.name"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.productType')">
        <el-select v-model="productAttr.productAttributeCategoryId" :placeholder="t('common.selectPlaceholder')">
          <el-option v-for="item in productAttrCateList" :key="item.id" :label="item.name" :value="item.id!">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item :label="t('product.filterStyle')">
        <el-radio-group v-model="productAttr.filterType">
          <el-radio :label="0">{{ t('mes.scheduling.priorityNORMAL') }}</el-radio>
          <el-radio :label="1">{{ t('product.color') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.searchable')">
        <el-radio-group v-model="productAttr.searchType">
          <el-radio :label="0">{{ t('product.noSearch') }}</el-radio>
          <el-radio :label="1">{{ t('product.keywordSearch') }}</el-radio>
          <el-radio :label="2">{{ t('product.rangeSearch') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.attrRelation')">
        <el-radio-group v-model="productAttr.relatedStatus">
          <el-radio :label="1">{{ t('role.page.yes') }}</el-radio>
          <el-radio :label="0">{{ t('role.page.no') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.attrOptional')">
        <el-radio-group v-model="productAttr.selectType">
          <el-radio :label="0">{{ t('product.unique') }}</el-radio>
          <el-radio :label="1">{{ t('product.singleSelect') }}</el-radio>
          <el-radio :label="2">{{ t('product.multiSelect') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.valueInput')">
        <el-radio-group v-model="productAttr.inputType">
          <el-radio :label="0">{{ t('product.manualInput') }}</el-radio>
          <el-radio :label="1">{{ t('product.selectFromList') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.valueOptions')">
        <el-input :autosize="true" type="textarea" v-model="inputListFormat"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.allowManualAdd')">
        <el-radio-group v-model="productAttr.handAddStatus">
          <el-radio :label="1">{{ t('role.page.yes') }}</el-radio>
          <el-radio :label="0">{{ t('role.page.no') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.sortAttr')">
        <el-input v-model="productAttr.sort"></el-input>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">{{ t('common.submit') }}</el-button>
        <el-button v-if="!isEdit" @click="resetForm">{{ t('common.reset') }}</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped></style>
