<script setup lang="ts">
import { ref, onMounted, provide, } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import ProductInfoDetail from './ProductInfoDetail.vue'
import ProductSaleDetail from './ProductSaleDetail.vue'
import ProductAttrDetail from './ProductAttrDetail.vue'
import ProductRelationDetail from './ProductRelationDetail.vue'
import { productCreateAPI, getPruductUpdateInfoAPI, productUpdateByIdAPI } from '@/apis/product'
import type { PmsProductParam } from '@/types/product'
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

// Default product parameters
const defaultProductParam: PmsProductParam = {
  albumPics: '',
  brandName: '',
  deleteStatus: 0,
  description: '',
  detailDesc: '',
  detailHtml: '',
  detailMobileHtml: '',
  detailTitle: '',
  feightTemplateId: 0,
  flashPromotionCount: 0,
  flashPromotionId: 0,
  flashPromotionPrice: 0,
  flashPromotionSort: 0,
  giftPoint: 0,
  giftGrowth: 0,
  keywords: '',
  lowStock: 0,
  name: '',
  newStatus: 0,
  note: '',
  originalPrice: 0,
  pic: '',
  memberPriceList: [],
  productFullReductionList: [{ fullPrice: 0, reducePrice: 0 }],
  productLadderList: [{ count: 0, discount: 0, price: 0 }],
  previewStatus: 0,
  price: 0,
  productAttributeValueList: [],
  skuStockList: [],
  subjectProductRelationList: [],
  prefrenceAreaProductRelationList: [],
  productCategoryName: '',
  productSn: '',
  promotionEndTime: '',
  promotionPerLimit: 0,
  promotionStartTime: '',
  promotionType: 0,
  publishStatus: 0,
  recommandStatus: 0,
  sale: 0,
  serviceIds: '',
  sort: 0,
  stock: 0,
  subTitle: '',
  unit: '',
  usePointLimit: 0,
  verifyStatus: 0,
  weight: 0
}
// el-steps active state
const active = ref(0)
// Display state of info/promo/attributes/relation components
const showStatus = ref([true, false, false, false])
// Current product parameters
const productParam = ref<PmsProductParam>(Object.assign({}, defaultProductParam))
// Implement cross-layer data passing
provide('product-key', productParam)

// Run after component mounts
onMounted(async () => {
  if (props.isEdit) {
    const res = await getPruductUpdateInfoAPI(Number(route.query.id))
    productParam.value = res.data
  }
})

// Hide all product edit components
const hideAll = () => {
  // Set all states to false
  showStatus.value.fill(false)
}

// Previous step
const prevStep = () => {
  if (active.value > 0 && active.value < showStatus.value.length) {
    active.value--
    hideAll()
    showStatus.value[active.value] = true
  }
}

// Next step
const nextStep = () => {
  if (active.value < showStatus.value.length - 1) {
    active.value++
    hideAll()
    showStatus.value[active.value] = true
  }
}

// Finish the step and submit data
const finishCommit = async (isEdit: boolean) => {
  await ElMessageBox.confirm(t('dialog.confirmSubmitProduct'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  if (isEdit) {
    await productUpdateByIdAPI(Number(route.query.id), productParam.value)
    ElMessage({
      type: 'success',
      message: '提交成功',
      duration: 1000
    })
    router.back()
  } else {
    await productCreateAPI(productParam.value)
    ElMessage({
      type: 'success',
      message: '提交成功',
      duration: 1000
    })
    location.reload()
  }
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-steps :active="active" finish-status="success" align-center>
      <el-step title="填写商品信息"></el-step>
      <el-step title="填写商品促销"></el-step>
      <el-step title="填写商品属性"></el-step>
      <el-step title="选择商品关联"></el-step>
    </el-steps>
    <product-info-detail v-show="showStatus[0]" :is-edit="isEdit" @next-step="nextStep">
    </product-info-detail>
    <product-sale-detail v-show="showStatus[1]" :is-edit="isEdit" @next-step="nextStep" @prev-step="prevStep">
    </product-sale-detail>
    <product-attr-detail v-show="showStatus[2]" v-model="productParam" :is-edit="isEdit" @next-step="nextStep"
      @prev-step="prevStep">
    </product-attr-detail>
    <product-relation-detail v-show="showStatus[3]" v-model="productParam" :is-edit="isEdit" @prev-step="prevStep"
      @finish-commit="finishCommit">
    </product-relation-detail>
  </el-card>
</template>

<style>
.form-container {
  width: 960px;
}

.form-inner-container {
  width: 800px;
}
</style>
