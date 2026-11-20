<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { getProductAttributeCategoryListAPI } from '@/apis/productAttrCate'
import { productAttributeCreateAPI, getProductAttributeByIdAPI, productAttributeUpdateAPI } from '@/apis/productAttr'
import type { PmsProductAttribute, PmsProductAttributeCategory } from '@/types/productAttr'

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
const onSubmit = async () => {
  if (!productAttrFrom.value) return
  const valid = await productAttrFrom.value.validate().catch(() => false)
  if (valid) {
    await ElMessageBox.confirm('是否提交数据', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    if (props.isEdit) {
      await productAttributeUpdateAPI(Number(route.query.id), productAttr.value)
      ElMessage.success('修改成功')
      router.back()
    } else {
      await productAttributeCreateAPI(productAttr.value)
      ElMessage.success('提交成功')
      resetForm()
    }
  } else {
    ElMessage.error('验证失败')
  }
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
      <el-form-item label="属性名称：" prop="name">
        <el-input v-model="productAttr.name"></el-input>
      </el-form-item>
      <el-form-item label="商品类型：">
        <el-select v-model="productAttr.productAttributeCategoryId" placeholder="请选择">
          <el-option v-for="item in productAttrCateList" :key="item.id" :label="item.name" :value="item.id!">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item label="分类筛选样式:">
        <el-radio-group v-model="productAttr.filterType">
          <el-radio :label="0">普通</el-radio>
          <el-radio :label="1">颜色</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="能否进行检索:">
        <el-radio-group v-model="productAttr.searchType">
          <el-radio :label="0">不需要检索</el-radio>
          <el-radio :label="1">关键字检索</el-radio>
          <el-radio :label="2">范围检索</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="商品属性关联:">
        <el-radio-group v-model="productAttr.relatedStatus">
          <el-radio :label="1">是</el-radio>
          <el-radio :label="0">否</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="属性是否可选:">
        <el-radio-group v-model="productAttr.selectType">
          <el-radio :label="0">唯一</el-radio>
          <el-radio :label="1">单选</el-radio>
          <el-radio :label="2">复选</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="属性值的录入方式:">
        <el-radio-group v-model="productAttr.inputType">
          <el-radio :label="0">手工录入</el-radio>
          <el-radio :label="1">从下面列表中选择</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="属性值可选值列表:">
        <el-input :autosize="true" type="textarea" v-model="inputListFormat"></el-input>
      </el-form-item>
      <el-form-item label="是否支持手动新增:">
        <el-radio-group v-model="productAttr.handAddStatus">
          <el-radio :label="1">是</el-radio>
          <el-radio :label="0">否</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="排序属性：">
        <el-input v-model="productAttr.sort"></el-input>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit">提交</el-button>
        <el-button v-if="!isEdit" @click="resetForm">重置</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped></style>
