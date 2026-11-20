<script lang="ts" setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { createBrandAPI, getBrandAPI, updateBrandAPI } from '@/apis/brand'
import SingleUpload from '@/components/Upload/singleUpload.vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import type { PmsBrand } from '@/types/brand'

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

// Handle brand form submission
const handleBrandSubmit = () => {
  brandFromRef.value!.validate(async (valid) => {
    if (valid) {
      await ElMessageBox.confirm('是否提交数据', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      })
      if (props.isEdit) {
        await updateBrandAPI(brandId.value, brand.value)
        brandFromRef.value!.resetFields()
        ElMessage({
          message: '修改成功',
          type: 'success',
          duration: 1000
        })
        router.back()
      } else {
        await createBrandAPI(brand.value)
        brandFromRef.value!.resetFields()
        brand.value = Object.assign({}, defaultBrand)
        ElMessage({
          message: '提交成功',
          type: 'success',
          duration: 1000
        })
      }
    } else {
      ElMessage({
        message: '验证失败',
        type: 'error',
        duration: 1000
      })
    }
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
      <el-form-item label="品牌名称：" prop="name">
        <el-input v-model="brand.name"></el-input>
      </el-form-item>
      <el-form-item label="品牌首字母：">
        <el-input v-model="brand.firstLetter"></el-input>
      </el-form-item>
      <el-form-item label="品牌LOGO：" prop="logo">
        <single-upload v-model="brand.logo"></single-upload>
      </el-form-item>
      <el-form-item label="品牌专区大图：">
        <single-upload v-model="brand.bigPic"></single-upload>
      </el-form-item>
      <el-form-item label="品牌故事：">
        <el-input placeholder="请输入内容" type="textarea" v-model="brand.brandStory" :autosize="true"></el-input>
      </el-form-item>
      <el-form-item label="排序：" prop="sort">
        <el-input v-model.number="brand.sort"></el-input>
      </el-form-item>
      <el-form-item label="是否显示：">
        <el-radio-group v-model="brand.showStatus">
          <el-radio :label="1">是</el-radio>
          <el-radio :label="0">否</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="品牌制造商：">
        <el-radio-group v-model="brand.factoryStatus">
          <el-radio :label="1">是</el-radio>
          <el-radio :label="0">否</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleBrandSubmit()">提交</el-button>
        <el-button v-if="!props.isEdit" @click="handleBrandReset()">重置</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style></style>
