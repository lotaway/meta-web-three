<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox, type FormRules } from 'element-plus'
import { couponCreateAPI, getCouponByIdAPI, couponUpdateByIdAPI } from '@/apis/coupon'
import { getProductListAPI } from '@/apis/product'
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate'
import { useRoute, useRouter } from 'vue-router'
import type { CouponSelectProductOptionVo, SmsCouponExt } from '@/types/coupon'
import { couponPlatforms, couponTypes } from '@/utils/constant'
import type { ElCascaderDataVo } from '@/types/common'

// Get route
const route = useRoute()
const router = useRouter()

// Define properties
const props = defineProps({
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Default coupon, used when creating
const defaultCoupon: SmsCouponExt = {
  type: 0,
  name: '',
  platform: 0,
  amount: 0,
  perLimit: 1,
  useType: 0,
  productRelationList: [],
  productCategoryRelationList: []
}
// Current coupon object
const coupon = ref(Object.assign({}, defaultCoupon))
// Product category info when usage type is "specific category"
const productCateOptions = ref<ElCascaderDataVo[]>([])
// Selected category IDs when coupon type is "specific category" ([parent, child])
const selectProductCate = ref<number[]>([])
// Get category list
const getProductCateList = async () => {
  try {
    const res = await getProductCategoryListWithChildrenAPI()
    const list = res.data
    productCateOptions.value = list.map(item => ({
      label: item.name,
      value: item.id!,
      children: item.children?.map(it => ({ label: it.name, value: it.id! }))
    }))
  } catch (error) {
    console.error('获取商品分类列表失败:', error)
  }
}

// Initialize data
onMounted(async () => {
  if (props.isEdit) {
    const response = await getCouponByIdAPI(Number(route.query.id))
    coupon.value = response.data
  }
  await getProductCateList()
})

// Table loading state when coupon type is "specific product"
const selectProductLoading = ref(false)
// Product options in the table when coupon type is "specific product"
const selectProductOptions = ref<CouponSelectProductOptionVo[]>([])
// Currently selected associated product ID when coupon type is "specific product"
const selectProduct = ref<number>()


// Form ref
const couponFrom = ref()
// Form validation rules
const rules = reactive<FormRules>({
  name: [
    { required: true, message: '请输入优惠券名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ],
  publishCount: [
    { type: 'number', required: true, message: '只能输入正整数', trigger: 'blur' }
  ],
  amount: [
    { type: 'number', required: true, message: '面值只能是数值，0.01-10000，限2位小数', trigger: 'blur' }
  ],
  minPoint: [
    { type: 'number', required: true, message: '只能输入正整数', trigger: 'blur' }
  ]
})

// Submit form
const onSubmit = async () => {
  if (!couponFrom.value) return
  const isValid = await couponFrom.value.validate().catch(() => false)
  if (isValid) {
    await ElMessageBox.confirm('是否提交数据', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    if (props.isEdit) {
      try {
        await couponUpdateByIdAPI(Number(route.query.id), coupon.value)
        couponFrom.value.resetFields()
        ElMessage({
          message: '修改成功',
          type: 'success',
          duration: 1000
        })
        router.back()
      } catch (error) {
        console.error('更新优惠券失败:', error)
      }
    } else {
      try {
        await couponCreateAPI(coupon.value)
        couponFrom.value.resetFields()
        ElMessage({
          message: '提交成功',
          type: 'success',
          duration: 1000
        })
        router.back()
      } catch (error) {
        console.error('创建优惠券失败:', error)
      }
    }
  } else {
    ElMessage({
      message: '验证失败',
      type: 'error',
      duration: 1000
    })
  }
}

// Reset form
const resetForm = () => {
  if (couponFrom.value) {
    couponFrom.value.resetFields()
  }
  coupon.value = Object.assign({}, defaultCoupon)
}

// Search products method
const searchProductMethod = async (query: string) => {
  if (query) {
    selectProductLoading.value = true
    try {
      const res = await getProductListAPI({ pageNum: 1, pageSize: 10, keyword: query })
      selectProductLoading.value = false
      const productList = res.data.list
      selectProductOptions.value = productList.map(item => ({
        productId: item.id,
        productName: item.name,
        productSn: item.productSn
      }))
    } catch (error) {
      selectProductLoading.value = false
      console.error('搜索商品失败:', error)
    }
  } else {
    selectProductOptions.value = []
  }
}

// Add product association
const handleAddProductRelation = () => {
  if (!selectProduct.value) {
    ElMessage({
      message: '请先选择商品',
      type: 'warning'
    })
    return
  }
  coupon.value.productRelationList!.push(getProductById(selectProduct.value)!)
  selectProduct.value = undefined
}

// Delete product association
const handleDeleteProductRelation = (index: number) => {
  coupon.value.productRelationList!.splice(index, 1)
}

// Add category association
const handleAddProductCategoryRelation = () => {
  if (selectProductCate.value.length <= 0) {
    ElMessage({
      message: '请先选择商品分类',
      type: 'warning'
    })
    return
  }
  coupon.value.productCategoryRelationList!.push(getProductCateByIds(selectProductCate.value))
  selectProductCate.value = []
}

// Delete category association
const handleDeleteProductCateRelation = (index: number) => {
  coupon.value.productCategoryRelationList!.splice(index, 1)
}

// Get associated product by ID
const getProductById = (id: number) => {
  return selectProductOptions.value.find(item => item.productId === id)
}

// Get product categories by IDs
const getProductCateByIds = (ids: number[]) => {
  const findParentCate = productCateOptions.value.find(item => item.value === ids[0])
  const findCate = findParentCate?.children?.find(item => item.value === ids[1])
  return { productCategoryId: ids[1], productCategoryName: findCate?.label, parentCategoryName: findParentCate?.label }
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-form :model="coupon" :rules="rules" ref="couponFrom" label-width="150px">
      <el-form-item label="优惠券类型：">
        <el-select v-model="coupon.type">
          <el-option v-for="type in couponTypes" :key="type.value" :label="type.label" :value="type.value">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item label="优惠券名称：" prop="name">
        <el-input v-model="coupon.name" class="input-width"></el-input>
      </el-form-item>
      <el-form-item label="适用平台：">
        <el-select v-model="coupon.platform">
          <el-option v-for="item in couponPlatforms" :key="item.value" :label="item.label" :value="item.value">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item label="总发行量：" prop="publishCount">
        <el-input v-model.number="coupon.publishCount" placeholder="只能输入正整数" class="input-width"></el-input>
      </el-form-item>
      <el-form-item label="面额：" prop="amount">
        <el-input v-model.number="coupon.amount" placeholder="面值只能是数值，限2位小数" class="input-width">
          <template #append>元</template>
        </el-input>
      </el-form-item>
      <el-form-item label="每人限领：">
        <el-input v-model="coupon.perLimit" placeholder="只能输入正整数" class="input-width">
          <template #append>张</template>
        </el-input>
      </el-form-item>
      <el-form-item label="使用门槛：" prop="minPoint">
        <el-input v-model.number="coupon.minPoint" placeholder="只能输入正整数" class="input-width">
          <template #prepend>满</template>
          <template #append>元可用</template>
        </el-input>
      </el-form-item>
      <el-form-item label="领取日期：" prop="enableTime">
        <el-date-picker type="date" placeholder="选择日期" v-model="coupon.enableTime" class="input-width"></el-date-picker>
      </el-form-item>
      <el-form-item label="有效期：">
        <el-date-picker type="date" placeholder="选择日期" v-model="coupon.startTime" style="width: 150px"></el-date-picker>
        <span style="margin-left: 20px;margin-right: 20px">至</span>
        <el-date-picker type="date" placeholder="选择日期" v-model="coupon.endTime" style="width: 150px"></el-date-picker>
      </el-form-item>
      <el-form-item label="可使用商品：">
        <el-radio-group v-model="coupon.useType">
          <el-radio-button :label="0">全场通用</el-radio-button>
          <el-radio-button :label="1">指定分类</el-radio-button>
          <el-radio-button :label="2">指定商品</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item v-show="coupon.useType === 1">
        <el-cascader clearable placeholder="请选择分类名称" v-model="selectProductCate" :options="productCateOptions">
        </el-cascader>
        <el-button @click="handleAddProductCategoryRelation()">添加</el-button>
        <el-table ref="productCateRelationTable" :data="coupon.productCategoryRelationList"
          style="width: 100%;margin-top: 20px" border>
          <el-table-column label="分类名称" align="center">
            <template #default="scope">{{ scope.row.parentCategoryName }}>{{ scope.row.productCategoryName }}</template>
          </el-table-column>
          <el-table-column label="操作" align="center" width="100">
            <template #default="scope">
              <el-button size="small" type="primary" link @click="handleDeleteProductCateRelation(scope.$index)">删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-form-item>
      <el-form-item v-show="coupon.useType === 2">
        <el-select v-model="selectProduct" filterable remote reserve-keyword placeholder="商品名称/商品货号"
          :remote-method="searchProductMethod" :loading="selectProductLoading" style="width: 300px;">
          <el-option v-for="item in selectProductOptions" :key="item.productId" :label="item.productName"
            :value="item.productId!">
            <template #default>
              <span style="float: left">{{ item.productName }}</span>
              <span style="float: right; color: #8492a6; font-size: 13px">NO.{{ item.productSn }}</span>
            </template>
          </el-option>
        </el-select>
        <el-button @click="handleAddProductRelation()">添加</el-button>
        <el-table ref="productRelationTable" :data="coupon.productRelationList" style="width: 100%;margin-top: 20px"
          border>
          <el-table-column label="商品名称" align="center">
            <template #default="scope">{{ scope.row.productName }}</template>
          </el-table-column>
          <el-table-column label="货号" align="center" width="120">
            <template #default="scope">NO.{{ scope.row.productSn }}</template>
          </el-table-column>
          <el-table-column label="操作" align="center" width="100">
            <template #default="scope">
              <el-button size="small" type="primary" link @click="handleDeleteProductRelation(scope.$index)">删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-form-item>
      <el-form-item label="备注：">
        <el-input class="input-width" type="textarea" :rows="5" placeholder="请输入内容" v-model="coupon.note">
        </el-input>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit()">提交</el-button>
        <el-button v-if="!isEdit" @click="resetForm()">重置</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped>
.input-width {
  width: 60%;
}
</style>
