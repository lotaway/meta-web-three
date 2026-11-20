<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import { ref, onMounted, watch, inject, type Ref } from 'vue'
import { ElMessage, type FormInstance } from 'element-plus'
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate'
import { getBrandListAPI } from '@/apis/brand'
import type { PmsProductParam } from '@/types/product'
import type { ElCascaderDataVo, ElSelectDataVo } from '@/types/common'

// Define properties
const props = defineProps({
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Define events
const emit = defineEmits(['next-step'])

// Get cross-layer passed data
const compProductParam = inject('product-key') as Ref<PmsProductParam>

// Currently selected product category in the cascade: [parent ID, category ID]
const selectProductCateValue = ref<number[]>([])
// Product category data in the cascade selector
const productCateOptions = ref<ElCascaderDataVo[]>([])
// Get product category list data
const getProductCateList = async () => {
  const res = await getProductCategoryListWithChildrenAPI()
  const list = res.data
  productCateOptions.value = list.map(item => ({
    label: item.name,
    value: item.id!,
    children: item.children?.map(it => ({ label: it.name, value: it.id! }))
  }))
}
// Brand data in the selector
const brandOptions = ref<ElSelectDataVo[]>([])
// Get brand list data
const getBrandList = async () => {
  const res = await getBrandListAPI({ pageNum: 1, pageSize: 100 })
  brandOptions.value = res.data.list.map(item => ({ label: item.name, value: item.id! }))
}

// Initialize data
onMounted(async () => {
  await getProductCateList()
  await getBrandList()
  if (props.isEdit) {
    handleEditCreated()
  }
})

// Edit-mode initialization flag
const hasEditCreated = ref(false)

// Form ref
const productInfoForm = ref<FormInstance>()
// Form validation rules
const rules = {
  name: [
    { required: true, message: '请输入商品名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ],
  subTitle: [{ required: true, message: '请输入商品副标题', trigger: 'blur' }],
  productCategoryId: [{ required: true, message: '请选择商品分类', trigger: 'blur' }],
  brandId: [{ required: true, message: '请选择商品品牌', trigger: 'blur' }],
  description: [{ required: true, message: '请输入商品介绍', trigger: 'blur' }],
  requiredProp: [{ required: true, message: '该项为必填项', trigger: 'blur' }]
}

// Watch category selection change
watch(selectProductCateValue, (newValue) => {
  if (newValue && newValue.length === 2) {
    compProductParam.value.productCategoryId = newValue[1]
    compProductParam.value.productCategoryName = getCateNameById(compProductParam.value.productCategoryId!)
  } else {
    compProductParam.value.productCategoryId = undefined
    compProductParam.value.productCategoryName = undefined
  }
})


// Method definitions
const handleEditCreated = () => {
  if (compProductParam.value.productCategoryId) {
    selectProductCateValue.value.push(compProductParam.value.cateParentId!)
    selectProductCateValue.value.push(compProductParam.value.productCategoryId)
  }
  hasEditCreated.value = true
}

// Get category name by ID
const getCateNameById = (id: number) => {
  for (const item of productCateOptions.value) {
    const child = item.children?.find(child => child.value === id)
    if (child) {
      return child.label
    }
  }
  return undefined
}

// Next-step button show event
const handleNext = async () => {
  if (productInfoForm.value) {
    try {
      const valid = await productInfoForm.value.validate()
      if (valid) {
        emit('next-step')
      }
    } catch {
      ElMessage({
        message: '验证失败',
        type: 'error',
        duration: 1000
      })
    }
  }
}

// Handle brand change
const handleBrandChange = (val: number) => {
  const findBrand = brandOptions.value.find(item => item.value === val)
  compProductParam.value.brandName = findBrand?.label
}
</script>

<template>
  <div style="margin-top: 50px">
    <el-form :model="compProductParam" :rules="rules" ref="productInfoForm" label-width="120px"
      class="form-inner-container">
      <el-form-item :label="t('product.productCategory')" prop="productCategoryId">
        <el-cascader v-model="selectProductCateValue" :options="productCateOptions">
        </el-cascader>
      </el-form-item>
      <el-form-item :label="t('product.productName')" prop="name">
        <el-input v-model="compProductParam.name"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.subtitle')" prop="subTitle">
        <el-input v-model="compProductParam.subTitle"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.productBrand')" prop="brandId">
        <el-select v-model="compProductParam.brandId" @change="handleBrandChange" :placeholder="t('product.selectBrand')">
          <el-option v-for="item in brandOptions" :key="item.value" :label="item.label" :value="item.value">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item :label="t('product.intro')">
        <el-input :autoSize="true" v-model="compProductParam.description" type="textarea"
          :placeholder="t('oms.order.list.notePlaceholder')"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.productSn')">
        <el-input v-model="compProductParam.productSn"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.price')">
        <el-input v-model="compProductParam.price"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.marketPrice')">
        <el-input v-model="compProductParam.originalPrice"></el-input>
      </el-form-item>
      <el-form-item :label="t('dialog.productStock')">
        <el-input v-model="compProductParam.stock"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.unit')">
        <el-input v-model="compProductParam.unit"></el-input>
      </el-form-item>
      <el-form-item :label="t('product.weight')">
        <el-input v-model="compProductParam.weight" style="width: 300px"></el-input>
        <span style="margin-left: 20px">{{ t('product.gram') }}</span>
      </el-form-item>
      <el-form-item :label="t('product.sort')">
        <el-input v-model="compProductParam.sort"></el-input>
      </el-form-item>
      <el-form-item>
        <div style="width: 100%;text-align: center;">
          <el-button type="primary" @click="handleNext()">{{ t('product.nextPromotion') }}</el-button>
        </div>
      </el-form-item>
    </el-form>
  </div>
</template>

<style scoped></style>
