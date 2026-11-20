<script setup lang="ts">
import { ref, reactive, computed, onMounted, inject, type Ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import { getProductAttributeCategoryListAPI as fetchProductAttrCateList } from '@/apis/productAttrCate'
import { getProductAttributeListAPI } from '@/apis/productAttr'
import SingleUpload from '@/components/Upload/singleUpload.vue'
import type { PmsProductParam } from '@/types/product'
import type { ElSelectDataVo } from '@/types/common'
import MultiUpload from '@/components/Upload/multiUpload.vue'
import type { PmsProductAttribute, ProductAttrPicVo, ProductAttrVo, ProductParamVo } from '@/types/productAttr'
import type { PmsSkuStock } from '@/types/skuStock'
import Tinymce from '@/components/Tinymce/index.vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Define properties
const props = defineProps({
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Define events
const emit = defineEmits(['prev-step', 'next-step'])

// Get cross-layer passed data
const compProductParam = inject('product-key') as Ref<PmsProductParam>

// Template ref
const productAttrForm = ref<FormInstance>()

// Reactive data
const state = reactive({
  // Whether initialization succeeded in edit mode
  hasEditCreated: false,
  // Product attribute category dropdown options
  productAttributeCategoryOptions: [] as ElSelectDataVo[],
  // Selected product specification
  selectProductAttr: [] as ProductAttrVo[],
  // Selected product parameter
  selectProductParam: [] as ProductParamVo[],
  // Selected product attribute image
  selectProductAttrPics: [] as ProductAttrPicVo[],
  // Manually addable product attributes
  addProductAttrValue: '',
  // Product rich-text detail active tab
  activeHtmlName: 'pc'
})

// Product ID
const productId = computed(() => {
  return compProductParam.value.id
})

// compProductParam loads async; watch it to initialize data
watch(productId, (newVal, oldVal) => {
  if (props.isEdit) {
    handleEditCreated()
  }
  console.log("attr", newVal, oldVal)
})

// Get product attribute categories
const getProductAttrCateList = async () => {
  const param = { pageNum: 1, pageSize: 100 }
  const res = await fetchProductAttrCateList(param)
  state.productAttributeCategoryOptions = res.data.list.map(item => ({
    label: item.name,
    value: item.id!
  }))
}

onMounted(async () => {
  await getProductAttrCateList()
})

// Whether there are product attribute images
const hasAttrPic = computed(() => {
  if (state.selectProductAttrPics.length < 1) {
    return false
  }
  return true
})


// Product main image and gallery images
const selectProductPics = computed({
  get: function () {
    const pics: string[] = []
    if (!compProductParam.value.pic || compProductParam.value.pic === '') {
      return pics
    }
    pics.push(compProductParam.value.pic)
    if (!compProductParam.value.albumPics || compProductParam.value.albumPics === '') {
      return pics
    }
    const albumPics = compProductParam.value.albumPics.split(',')
    pics.push(...albumPics)
    return pics
  },
  set: function (newValue) {
    if (!newValue || newValue.length === 0) {
      compProductParam.value.pic = undefined
      compProductParam.value.albumPics = undefined
    } else {
      compProductParam.value.pic = newValue[0]
      compProductParam.value.albumPics = ''
      if (newValue.length > 1) {
        for (let i = 1; i < newValue.length; i++) {
          compProductParam.value.albumPics += newValue[i]
          if (i !== newValue.length - 1) {
            compProductParam.value.albumPics += ','
          }
        }
      }
    }
  }
})


// Handle data initialized in edit mode
const handleEditCreated = () => {
  if (compProductParam.value.productAttributeCategoryId) {
    handleProductAttrChange(compProductParam.value.productAttributeCategoryId)
  }
  state.hasEditCreated = true
}


const buildSelectProductAttr = (list: PmsProductAttribute[]): ProductAttrVo[] =>
  list.map((item, i) => {
    let options: string[] | undefined = []
    let values: string[] = []
    if (props.isEdit) {
      if (item.handAddStatus === 1) {
        options = getEditAttrOptions(item.id!)
      }
      values = getEditAttrValues(i)
    }
    return {
      id: item.id,
      name: item.name,
      handAddStatus: item.handAddStatus,
      inputList: item.inputList,
      values,
      options
    }
  })

const buildSelectProductParam = (list: PmsProductAttribute[]): ProductParamVo[] =>
  list.map(item => ({
    id: item.id,
    name: item.name,
    value: props.isEdit ? getEditParamValue(item.id!) : undefined,
    inputType: item.inputType,
    inputList: item.inputList
  }))

/**
 * Get specifications and parameters by product attribute category ID
 * @param type 0: specification; 1: parameter
 * @param cid product attribute category ID
 */
const getProductAttrList = async (type: number, cid: number) => {
  const param = { pageNum: 1, pageSize: 100, type: type }
  const res = await getProductAttributeListAPI(cid, param)
  const list = res.data.list
  if (type === 0) {
    state.selectProductAttr = buildSelectProductAttr(list)
    if (props.isEdit) {
      refreshProductAttrPics()
    }
  } else {
    state.selectProductParam = buildSelectProductParam(list)
  }
}

// Get configured manually addable attribute values
const getEditAttrOptions = (id: number) => {
  const attrValueList = compProductParam.value.productAttributeValueList
  return attrValueList?.find(item => item.productAttributeId === id)?.value?.split(',')
}

// Get selected specification values
const getEditAttrValues = (index: number) => {
  const skuList = compProductParam.value.skuStockList!
  const values = new Set<string>()
  for (let i = 0; i < skuList.length; i++) {
    const sku = skuList[i]
    const spData = JSON.parse(sku!.spData!)
    if (spData && spData.length > index) {
      values.add(spData[index].value)
    }
  }
  return Array.from(values)
}

// Get attribute values
const getEditParamValue = (id: number) => {
  return compProductParam.value.productAttributeValueList?.find(item => item.productAttributeId === id)?.value
}

const handleProductAttrChange = (value: number) => {
  getProductAttrList(0, value)
  getProductAttrList(1, value)
}

const getInputListArr = (inputList: string) => {
  return inputList.split(',')
}

const showAttrWarning = (message: string) => {
  ElMessage({ message, type: 'warning', duration: 1000 })
}

const handleAddProductAttrValue = (idx: number) => {
  const options = state.selectProductAttr[idx]!.options
  if (state.addProductAttrValue == null || state.addProductAttrValue == '') {
    showAttrWarning('属性值不能为空')
    return
  }
  if (options!.indexOf(state.addProductAttrValue) !== -1) {
    showAttrWarning('属性值不能重复')
    return
  }
  options?.push(state.addProductAttrValue)
  state.addProductAttrValue = ''
}

const handleRemoveProductAttrValue = (idx: number, index: number) => {
  state.selectProductAttr[idx]?.options?.splice(index, 1)
}

const getProductSkuSp = (row: PmsSkuStock, index: number) => {
  const spData = JSON.parse(row.spData!)
  if (spData != null && index < spData.length) {
    return spData[index].value
  } else {
    return null
  }
}

const handleRefreshProductSkuList = async () => {
  try {
    await ElMessageBox.confirm(t('product.confirmRefreshSku'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    refreshProductAttrPics()
    refreshProductSkuList()
  } catch {
    // User cancelled the operation
  }
}

const syncSkuPrice = () => {
  const skuStockList = compProductParam.value.skuStockList
  if (!skuStockList || skuStockList.length === 0) return
  let tempSkuList: PmsSkuStock[] = []
  tempSkuList = tempSkuList.concat(tempSkuList, skuStockList)
  const price = skuStockList[0]!.price
  for (let i = 0; i < tempSkuList.length; i++) {
    tempSkuList[i]!.price = price
  }
  compProductParam.value.skuStockList = []
  compProductParam.value.skuStockList = compProductParam.value.skuStockList.concat(compProductParam.value.skuStockList, tempSkuList)
}

const syncSkuStock = () => {
  const skuStockList = compProductParam.value.skuStockList
  if (!skuStockList || skuStockList.length === 0) return
  let tempSkuList: PmsSkuStock[] = []
  tempSkuList = tempSkuList.concat(tempSkuList, skuStockList)
  const stock = skuStockList[0]!.stock
  const lowStock = skuStockList[0]!.lowStock
  for (let i = 0; i < tempSkuList.length; i++) {
    tempSkuList[i]!.stock = stock
    tempSkuList[i]!.lowStock = lowStock
  }
  compProductParam.value.skuStockList = []
  compProductParam.value.skuStockList = compProductParam.value.skuStockList.concat(compProductParam.value.skuStockList, tempSkuList)
}

const handleSyncProductSkuPrice = async () => {
  try {
    await ElMessageBox.confirm(t('product.confirmSyncPrice'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    syncSkuPrice()
  } catch {
    // User cancelled the operation
  }
}

const handleSyncProductSkuStock = async () => {
  try {
    await ElMessageBox.confirm(t('product.confirmSyncStock'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    syncSkuStock()
  } catch {
    // User cancelled the operation
  }
}

type SkuSpPair = { key: string; value: string | undefined }

const makeSkuPair = (attr: ProductAttrVo, index: number): SkuSpPair => ({
  key: attr.name,
  value: attr.values![index]
})

const pushSkuCombination = (skuList: PmsSkuStock[], pairs: SkuSpPair[]) => {
  skuList.push({ spData: JSON.stringify(pairs) })
}

const fillSkuListForOneAttr = (skuList: PmsSkuStock[], attr0: ProductAttrVo) => {
  for (let i = 0; i < attr0.values!.length; i++) {
    pushSkuCombination(skuList, [makeSkuPair(attr0, i)])
  }
}

const fillSkuListForTwoAttrs = (skuList: PmsSkuStock[], attr0: ProductAttrVo, attr1: ProductAttrVo) => {
  for (let i = 0; i < attr0.values!.length; i++) {
    if (attr1.values!.length === 0) {
      pushSkuCombination(skuList, [makeSkuPair(attr0, i)])
      continue
    }
    for (let j = 0; j < attr1.values!.length; j++) {
      pushSkuCombination(skuList, [makeSkuPair(attr0, i), makeSkuPair(attr1, j)])
    }
  }
}

const appendThirdAttrSkus = (skuList: PmsSkuStock[], pairs: SkuSpPair[], attr2: ProductAttrVo) => {
  if (attr2.values!.length === 0) {
    pushSkuCombination(skuList, pairs)
    return
  }
  for (let k = 0; k < attr2.values!.length; k++) {
    pushSkuCombination(skuList, [...pairs, makeSkuPair(attr2, k)])
  }
}

const fillSkuListForThreeAttrs = (
  skuList: PmsSkuStock[],
  attr0: ProductAttrVo,
  attr1: ProductAttrVo,
  attr2: ProductAttrVo
) => {
  for (let i = 0; i < attr0.values!.length; i++) {
    if (attr1.values!.length === 0) {
      pushSkuCombination(skuList, [makeSkuPair(attr0, i)])
      continue
    }
    for (let j = 0; j < attr1.values!.length; j++) {
      appendThirdAttrSkus(skuList, [makeSkuPair(attr0, i), makeSkuPair(attr1, j)], attr2)
    }
  }
}

const refreshProductSkuList = () => {
  compProductParam.value.skuStockList = []
  const skuList = compProductParam.value.skuStockList
  const attrs = state.selectProductAttr
  if (attrs.length === 1) {
    fillSkuListForOneAttr(skuList, attrs[0]!)
  } else if (attrs.length === 2) {
    fillSkuListForTwoAttrs(skuList, attrs[0]!, attrs[1]!)
  } else {
    fillSkuListForThreeAttrs(skuList, attrs[0]!, attrs[1]!, attrs[2]!)
  }
}

const refreshProductAttrPics = () => {
  state.selectProductAttrPics = []
  if (state.selectProductAttr && state.selectProductAttr.length >= 1) {
    const values = state.selectProductAttr[0]!.values!
    for (let i = 0; i < values.length; i++) {
      let pic = undefined
      if (props.isEdit) {
        // Get images in edit mode
        pic = getProductSkuPic(values[i]!)
      }
      state.selectProductAttrPics.push({ name: values[i], pic: pic })
    }
  }
}

// Get images of product-related attributes
const getProductSkuPic = (name: string) => {
  return compProductParam.value.skuStockList?.find(item => {
    const spData = JSON.parse(item.spData!)
    return name === spData[0].value
  })?.pic
}

// Merge product attributes
const mergeProductAttrValue = () => {
  compProductParam.value.productAttributeValueList = []
  for (let i = 0; i < state.selectProductAttr.length; i++) {
    const attr = state.selectProductAttr[i]!
    if (attr.handAddStatus === 1 && attr.options != null && attr.options.length > 0) {
      compProductParam.value.productAttributeValueList.push({
        productAttributeId: attr.id,
        value: getOptionStr(attr.options)
      })
    }
  }
  for (let i = 0; i < state.selectProductParam.length; i++) {
    const param = state.selectProductParam[i]!
    compProductParam.value.productAttributeValueList.push({
      productAttributeId: param.id,
      value: param.value
    })
  }
}

// Merge product attribute images
const mergeProductAttrPics = () => {
  for (let i = 0; i < state.selectProductAttrPics.length; i++) {
    const skuStockList = compProductParam.value.skuStockList!
    for (let j = 0; j < skuStockList.length; j++) {
      const spData = JSON.parse(skuStockList[j]!.spData!)
      if (spData[0].value === state.selectProductAttrPics[i]!.name) {
        skuStockList[j]!.pic = state.selectProductAttrPics[i]!.pic
      }
    }
  }
}

const getOptionStr = (arr: string[]) => {
  let str = ''
  for (let i = 0; i < arr.length; i++) {
    str += arr[i]
    if (i != arr.length - 1) {
      str += ','
    }
  }
  return str
}

const handleRemoveProductSku = (index: number) => {
  const list = compProductParam.value.skuStockList!
  if (list.length === 1) {
    list.pop()
  } else {
    list.splice(index, 1)
  }
}

const getParamInputList = (inputList: string) => {
  return inputList.split(',')
}

const handlePrev = () => {
  emit('prev-step')
}

const handleNext = () => {
  mergeProductAttrValue()
  mergeProductAttrPics()
  emit('next-step')
}
</script>

<template>
  <div style="margin-top: 50px">
    <el-form :model="compProductParam" ref="productAttrForm" label-width="120px" class="form-inner-container">
      <el-form-item :label="t('product.attrType')">
        <el-select v-model="compProductParam.productAttributeCategoryId" :placeholder="t('product.selectAttrType')"
          @change="handleProductAttrChange">
          <el-option v-for="item in state.productAttributeCategoryOptions" :key="item.value" :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item :label="t('product.specification')">
        <el-card shadow="never" class="cardBg">
          <div v-for="(productAttr, idx) in state.selectProductAttr" :key="'productAttr' + idx">
            {{ productAttr.name }}：
            <el-checkbox-group v-if="productAttr.handAddStatus === 0" v-model="state.selectProductAttr[idx]!.values">
              <el-checkbox v-for="item in getInputListArr(productAttr.inputList!)" :label="item" :key="item"
                class="littleMarginLeft"></el-checkbox>
            </el-checkbox-group>
            <div v-else>
              <el-checkbox-group v-model="state.selectProductAttr[idx]!.values">
                <div v-for="(item, index) in state.selectProductAttr[idx]!.options" style="display: inline-flex;"
                  class="littleMarginLeft" :key="'optons' + index">
                  <el-checkbox :label="item" :key="item"></el-checkbox>
                  <el-button type="primary" link class="littleMarginLeft"
                    @click="handleRemoveProductAttrValue(idx, index)">{{ t('common.delete') }}
                  </el-button>
                </div>
              </el-checkbox-group>
              <el-input v-model="state.addProductAttrValue" style="width: 160px;margin-left: 10px" clearable></el-input>
              <el-button class="littleMarginLeft" @click="handleAddProductAttrValue(idx)">{{ t('product.increase') }}</el-button>
            </div>
          </div>
        </el-card>
        <el-table style="width: 100%;margin-top: 20px" :data="compProductParam.skuStockList" border>
          <el-table-column v-for="(item, index) in state.selectProductAttr" :label="item.name" :key="item.id"
            align="center">
            <template #default="scope">
              {{ getProductSkuSp(scope.row, index) }}
            </template>
          </el-table-column>
          <el-table-column label="销售价格" width="100" align="center">
            <template #default="scope">
              <el-input v-model="scope.row.price"></el-input>
            </template>
          </el-table-column>
          <el-table-column label="促销价格" width="100" align="center">
            <template #default="scope">
              <el-input v-model="scope.row.promotionPrice"></el-input>
            </template>
          </el-table-column>
          <el-table-column label="商品库存" width="80" align="center">
            <template #default="scope">
              <el-input v-model="scope.row.stock"></el-input>
            </template>
          </el-table-column>
          <el-table-column label="库存预警值" width="80" align="center">
            <template #default="scope">
              <el-input v-model="scope.row.lowStock"></el-input>
            </template>
          </el-table-column>
          <el-table-column label="SKU编号" width="160" align="center">
            <template #default="scope">
              <el-input v-model="scope.row.skuCode"></el-input>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80" align="center">
            <template #default="scope">
              <el-button type="primary" link @click="handleRemoveProductSku(scope.$index)">{{ t('common.delete') }}
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <el-button type="primary" style="margin-top: 20px" @click="handleRefreshProductSkuList">刷新列表
        </el-button>
        <el-button type="primary" style="margin-top: 20px" @click="handleSyncProductSkuPrice">同步价格
        </el-button>
        <el-button type="primary" style="margin-top: 20px" @click="handleSyncProductSkuStock">同步库存
        </el-button>
      </el-form-item>
      <el-form-item label="属性图片：" v-if="hasAttrPic">
        <el-card shadow="never" class="cardBg">
          <div v-for="(item, index) in state.selectProductAttrPics" :key="'productAttrPic' + index">
            <span>{{ item.name }}:</span>
            <single-upload v-model="item.pic"
              style="width: 300px;display: inline-block;margin-left: 10px"></single-upload>
          </div>
        </el-card>
      </el-form-item>
      <el-form-item label="商品参数：">
        <el-card shadow="never" class="cardBg">
          <div v-for="(item, index) in state.selectProductParam" :class="{ littleMarginTop: index !== 0 }"
            :key="'productParam' + index">
            <div class="paramInputLabel">{{ item.name }}:</div>
            <el-select v-if="item.inputType === 1" class="paramInput" v-model="state.selectProductParam[index]!.value">
              <el-option v-for="item2 in getParamInputList(item.inputList!)" :key="item2" :label="item2" :value="item2">
              </el-option>
            </el-select>
            <el-input v-else class="paramInput" v-model="state.selectProductParam[index]!.value"></el-input>
          </div>
        </el-card>
      </el-form-item>
      <el-form-item label="商品相册：">
        <multi-upload v-model="selectProductPics"></multi-upload>
      </el-form-item>
      <el-form-item label="商品详情：">
        <el-tabs v-model="state.activeHtmlName" type="card" style="width: 100%;">
          <el-tab-pane label="电脑端详情" name="pc">
            <tinymce :height="720" v-model="compProductParam.detailHtml!" />
          </el-tab-pane>
          <el-tab-pane label="移动端详情" name="mobile">
            <tinymce :height="720" v-model="compProductParam.detailMobileHtml!" />
          </el-tab-pane>
        </el-tabs>
      </el-form-item>
      <el-form-item>
        <div style="width: 100%; text-align: center;">
          <el-button @click="handlePrev">上一步，填写商品促销</el-button>
          <el-button type="primary" @click="handleNext">下一步，选择商品关联</el-button>
        </div>
      </el-form-item>
    </el-form>
  </div>
</template>

<style scoped>
.littleMarginLeft {
  margin-left: 10px;
}

.littleMarginTop {
  margin-top: 10px;
}

.paramInput {
  width: 250px;
}

.paramInputLabel {
  display: inline-block;
  width: 100px;
  text-align: right;
  padding-right: 10px
}

.cardBg {
  background: #F8F9FC;
}
</style>
