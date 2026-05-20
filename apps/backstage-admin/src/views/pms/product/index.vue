<script setup lang="ts">
import { ref, onMounted, reactive, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  getProductListAPI,
  productUpdateDeleteStatusAPI,
  productUpdateNewStatusAPI,
  productUpdateRecommendStatusAPI,
  productUpdatePublishStatusAPI
} from '@/apis/product'
import { getSkuListByPidAPI, skuUpdateByPidAPI } from '@/apis/skuStock'
import { getProductAttributeListAPI } from '@/apis/productAttr'
import { getBrandListAPI } from '@/apis/brand'
import { getProductCategoryListWithChildrenAPI } from '@/apis/productCate'
import { Search, Tickets, Edit } from '@element-plus/icons-vue'
import type { PmsProduct, ProductQueryParam } from '@/types/product'
import type { ElCascaderDataVo, ElSelectDataVo } from '@/types/common'
import type { PmsSkuStock } from '@/types/skuStock'
import type { PmsProductAttribute } from '@/types/productAttr'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

const router = useRouter()

const listQuery = ref<ProductQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE
})
const list = ref<PmsProduct[]>([])
const total = ref(0)
const listLoading = ref(true)
const getList = async () => {
  listLoading.value = true
  try {
    const response = await getProductListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data.list
    total.value = response.data.total
  } catch {
    listLoading.value = false
  }
}
const brandOptions = ref<ElSelectDataVo[]>([])
const getBrandList = async () => {
  const res = await getBrandListAPI({ pageNum: 1, pageSize: 100 })
  brandOptions.value = res.data.list.map(item => ({ label: item.name, value: item.id!.toString() }))
}
const productCateOptions = ref<ElCascaderDataVo[]>([])
const selectProductCateValue = ref([])
const getProductCateList = async () => {
  const res = await getProductCategoryListWithChildrenAPI()
  const list = res.data
  productCateOptions.value = list.map(item => ({
    label: item.name,
    value: item.id!,
    children: item.children?.map(it => ({ label: it.name, value: it.id! }))
  }))
}
const publishStatusOptions = ref([
  { value: 1, label: t('product.onSale') },
  { value: 0, label: t('product.offSale') }
])
const verifyStatusOptions = ref([
  { value: 1, label: t('product.verifyPassed') },
  { value: 0, label: t('product.notVerified') }
])

watch(selectProductCateValue, (newValue) => {
  if (newValue != null && newValue.length == 2) {
    listQuery.value.productCategoryId = newValue[1]
  } else {
    listQuery.value.productCategoryId = undefined
  }
}, { immediate: true })

onMounted(() => {
  getList()
  getBrandList()
  getProductCateList()
})

const operates = ref([
  { label: t('product.productOnSale'), value: 'publishOn' },
  { label: t('product.productOffSale'), value: 'publishOff' },
  { label: t('product.setRecommend'), value: 'recommendOn' },
  { label: t('product.cancelRecommend'), value: 'recommendOff' },
  { label: t('product.setNew'), value: 'newOn' },
  { label: t('product.cancelNew'), value: 'newOff' },
  { label: t('product.moveToRecycleBin'), value: 'recycle' },
  { label: t('product.transferCategory'), value: 'transferCategory' }
])
const operateType = ref<string>()
const multipleSelection = ref<PmsProduct[]>([])

const editSkuInfo = reactive({
  dialogVisible: false,
  productId: 0,
  productSn: '',
  productAttributeCategoryId: 0,
  stockList: [] as PmsSkuStock[],
  productAttr: [] as PmsProductAttribute[],
  keyword: undefined
})

const getProductSkuSp = (row: PmsSkuStock, index: number) => {
  const spData = JSON.parse(row.spData!)
  if (spData && index < spData.length) {
    return spData[index].value
  } else {
    return ''
  }
}

const handleShowSkuEditDialog = async (index: number, row: PmsProduct) => {
  editSkuInfo.dialogVisible = true
  editSkuInfo.productId = row.id!
  editSkuInfo.productSn = row.productSn
  editSkuInfo.productAttributeCategoryId = row.productAttributeCategoryId!
  editSkuInfo.keyword = undefined
  const resp = await getSkuListByPidAPI(row.id!, { keyword: editSkuInfo.keyword })
  editSkuInfo.stockList = resp.data
  if (row.productAttributeCategoryId) {
    const res2 = await getProductAttributeListAPI(row.productAttributeCategoryId, { pageNum: 1, pageSize: 10, type: 0 })
    editSkuInfo.productAttr = res2.data.list
  }
}

const handleSearchEditSku = async () => {
  const response = await getSkuListByPidAPI(editSkuInfo.productId, { keyword: editSkuInfo.keyword })
  editSkuInfo.stockList = response.data
}

const handleEditSkuConfirm = async () => {
  if (!editSkuInfo.stockList || editSkuInfo.stockList.length <= 0) {
    ElMessage({
      message: t('message.noSkuInfo'),
      type: 'warning',
      duration: MESSAGE_DURATION_SHORT
    })
    return
  }
  await ElMessageBox.confirm(t('message.confirmModify'), t('common.warning'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await skuUpdateByPidAPI(editSkuInfo.productId, editSkuInfo.stockList)
  ElMessage({
    message: t('message.modifySuccess'),
    type: 'success',
    duration: MESSAGE_DURATION_SHORT
  })
  editSkuInfo.dialogVisible = false
}

const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

const handleAddProduct = () => {
  router.push({ path: '/pms/addProduct' })
}

const handleBatchOperate = async () => {
  if (!operateType.value) {
    ElMessage({
      message: t('message.selectOperationType'),
      type: 'warning',
      duration: MESSAGE_DURATION_SHORT
    })
    return
  }
  if (!multipleSelection.value || multipleSelection.value.length < 1) {
    ElMessage({
      message: t('message.selectOperateProduct'),
      type: 'warning',
      duration: MESSAGE_DURATION_SHORT
    })
    return
  }
  await ElMessageBox.confirm(t('message.confirmBatchOperation'), t('common.warning'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  const ids = multipleSelection.value.map(item => item.id!)
  switch (operateType.value) {
    case operates.value[0]!.value:
      updatePublishStatus(1, ids)
      break
    case operates.value[1]!.value:
      updatePublishStatus(0, ids)
      break
    case operates.value[2]!.value:
      updateRecommendStatus(1, ids)
      break
    case operates.value[3]!.value:
      updateRecommendStatus(0, ids)
      break
    case operates.value[4]!.value:
      updateNewStatus(1, ids)
      break
    case operates.value[5]!.value:
      updateNewStatus(0, ids)
      break
    case operates.value[6]!.value:
      updateDeleteStatus(1, ids)
      break
    case operates.value[7]!.value:
      break
    default:
      break
  }
  getList()
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

const handleSelectionChange = (val: PmsProduct[]) => {
  multipleSelection.value = val
}

const handlePublishStatusChange = async (index: number, row: PmsProduct) => {
  await updatePublishStatus(row.publishStatus!, [row.id!])
}

const handleNewStatusChange = async (index: number, row: PmsProduct) => {
  await updateNewStatus(row.newStatus!, [row.id!])
}

const handleRecommendStatusChange = async (index: number, row: PmsProduct) => {
  await updateRecommendStatus(row.recommandStatus!, [row.id!])
}

const handleResetSearch = () => {
  selectProductCateValue.value = []
  listQuery.value = { pageNum: 1, pageSize: DEFAULT_PAGE_SIZE }
}

const handleDelete = async (index: number, row: PmsProduct) => {
  updateDeleteStatus(1, [row.id!])
}

const handleUpdateProduct = (index: number, row: PmsProduct) => {
  router.push({ path: '/pms/updateProduct', query: { id: row.id } })
}

const handleShowProduct = (index: number, row: PmsProduct) => {
}

const handleShowVerifyDetail = (index: number, row: PmsProduct) => {
}

const handleShowLog = (index: number, row: PmsProduct) => {
}

const updatePublishStatus = async (publishStatus: number, ids: number[]) => {
  await productUpdatePublishStatusAPI({ ids: ids.join(','), publishStatus: publishStatus })
  ElMessage({
    message: t('message.modifySuccess'),
    type: 'success',
    duration: MESSAGE_DURATION_SHORT
  })
}

const updateNewStatus = async (newStatus: number, ids: number[]) => {
  await productUpdateNewStatusAPI({ ids: ids.join(','), newStatus: newStatus })
  ElMessage({
    message: t('message.modifySuccess'),
    type: 'success',
    duration: MESSAGE_DURATION_SHORT
  })
}

const updateRecommendStatus = async (recommendStatus: number, ids: number[]) => {
  await productUpdateRecommendStatusAPI({ ids: ids.join(','), recommendStatus: recommendStatus })
  ElMessage({
    message: t('message.modifySuccess'),
    type: 'success',
    duration: MESSAGE_DURATION_SHORT
  })
}

const updateDeleteStatus = async (deleteStatus: number, ids: number[]) => {
  await productUpdateDeleteStatusAPI({ ids: ids.join(','), deleteStatus: deleteStatus })
  ElMessage({
    message: t('message.deleteSuccess'),
    type: 'success',
    duration: MESSAGE_DURATION_SHORT
  })
  getList()
}

const verifyStatusFilter = (value: number) => {
  if (value === 1) {
    return t('product.verifyPassed')
  } else {
    return t('product.notVerified')
  }
}
</script>

<template>
  <div class="app-container">
    <el-card class="filter-container" shadow="never">
      <div>
        <el-icon class="el-icon-middle">
          <Search />
        </el-icon>
        <span>{{ t('product.filterSearch') }}</span>
        <el-button style="float: right" @click="handleSearchList()" type="primary">
          {{ t('product.queryResult') }}
        </el-button>
        <el-button style="float: right;margin-right: 15px" @click="handleResetSearch()">
          {{ t('product.reset') }}
        </el-button>
      </div>
      <div style="margin-top: 20px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item :label="t('product.productName') + '：'">
            <el-input style="width: 203px" v-model="listQuery.keyword" :placeholder="t('product.productName')"></el-input>
          </el-form-item>
          <el-form-item :label="t('product.productSn') + '：'">
            <el-input style="width: 203px" v-model="listQuery.productSn" :placeholder="t('product.productSn')"></el-input>
          </el-form-item>
          <el-form-item :label="t('product.productCategory') + '：'">
            <el-cascader clearable v-model="selectProductCateValue" :options="productCateOptions">
            </el-cascader>
          </el-form-item>
          <el-form-item :label="t('product.brand') + '：'">
            <el-select v-model="listQuery.brandId" :placeholder="t('product.selectBrand')" clearable style="width: 203px;">
              <el-option v-for="item in brandOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item :label="t('product.onSale') + '：'">
            <el-select v-model="listQuery.publishStatus" :placeholder="t('product.all')" clearable style="width: 203px;">
              <el-option v-for="item in publishStatusOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item :label="t('product.verifyStatus') + '：'">
            <el-select v-model="listQuery.verifyStatus" :placeholder="t('product.all')" clearable style="width: 203px;">
              <el-option v-for="item in verifyStatusOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('product.dataList') }}</span>
      <el-button class="btn-add" @click="handleAddProduct()">
        {{ t('common.add') }}
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="productTable" :data="list" style="width: 100%" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column :label="t('product.productImage')" width="120" align="center">
          <template #default="scope"><img style="height: 80px" :src="scope.row.pic" :alt="scope.row.name"></template>
        </el-table-column>
        <el-table-column :label="t('product.productName')" align="center">
          <template #default="scope">
            <p>{{ scope.row.name }}</p>
            <p>{{ t('product.brand') }}：{{ scope.row.brandName }}</p>
          </template>
        </el-table-column>
        <el-table-column :label="t('product.priceSn')" width="120" align="center">
          <template #default="scope">
            <p>{{ t('product.price') }}：￥{{ scope.row.price }}</p>
            <p>{{ t('product.sn') }}：{{ scope.row.productSn }}</p>
          </template>
        </el-table-column>
        <el-table-column :label="t('product.tag')" width="140" align="center">
          <template #default="scope">
            <p style="margin: 6px 0px;">{{ t('product.onSale') }}：
              <el-switch @change="handlePublishStatusChange(scope.$index, scope.row)" :active-value="1"
                :inactive-value="0" v-model="scope.row.publishStatus">
              </el-switch>
            </p>
            <p style="margin: 6px 0px;">{{ t('product.new') }}：
              <el-switch @change="handleNewStatusChange(scope.$index, scope.row)" :active-value="1" :inactive-value="0"
                v-model="scope.row.newStatus">
              </el-switch>
            </p>
            <p style="margin: 6px 0px;">{{ t('product.recommended') }}：
              <el-switch @change="handleRecommendStatusChange(scope.$index, scope.row)" :active-value="1"
                :inactive-value="0" v-model="scope.row.recommandStatus">
              </el-switch>
            </p>
          </template>
        </el-table-column>
        <el-table-column :label="t('product.sort')" width="100" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column :label="t('product.skuStock')" width="100" align="center">
          <template #default="scope">
            <el-button type="primary" :icon="Edit" size="large"
              @click="handleShowSkuEditDialog(scope.$index, scope.row)" circle></el-button>
          </template>
        </el-table-column>
        <el-table-column :label="t('product.sales')" width="100" align="center">
          <template #default="scope">{{ scope.row.sale }}</template>
        </el-table-column>
        <el-table-column :label="t('product.verifyStatus')" width="100" align="center">
          <template #default="scope">
            <p>{{ verifyStatusFilter(scope.row.verifyStatus) }}</p>
            <p>
              <el-button type="primary" link @click="handleShowVerifyDetail(scope.$index, scope.row)">
                {{ t('product.verifyDetail') }}
              </el-button>
            </p>
          </template>
        </el-table-column>
        <el-table-column :label="t('product.operation')" width="160" align="center">
          <template #default="scope">
            <p>
              <el-button size="small" @click="handleShowProduct(scope.$index, scope.row)">
                {{ t('product.view') }}
              </el-button>
              <el-button size="small" @click="handleUpdateProduct(scope.$index, scope.row)">
                {{ t('common.edit') }}
              </el-button>
            </p>
            <p>
              <el-button size="small" @click="handleShowLog(scope.$index, scope.row)">
                {{ t('product.log') }}
              </el-button>
              <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row)">
                {{ t('common.delete') }}
              </el-button>
            </p>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" :placeholder="t('product.batchOperate')">
        <el-option v-for="item in operates" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        {{ t('common.confirm') }}
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="PAGE_SIZE_OPTIONS"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
    <el-dialog :title="t('dialog.editProductInfo')" v-model="editSkuInfo.dialogVisible" width="40%">
      <span>{{ t('dialog.productSn') }}：</span>
      <span>{{ editSkuInfo.productSn }}</span>
      <el-input :placeholder="t('dialog.searchBySkuCode')" v-model="editSkuInfo.keyword" style="width: 60%;margin-left: 20px">
        <template #append>
          <el-button :icon="Search" @click="handleSearchEditSku"></el-button>
        </template>
      </el-input>
      <el-table style="width: 100%;margin-top: 20px" :data="editSkuInfo.stockList" border>
        <el-table-column :label="t('dialog.skuCode')" align="center">
          <template #default="scope">
            <el-input v-model="scope.row.skuCode"></el-input>
          </template>
        </el-table-column>
        <el-table-column v-for="(item, index) in editSkuInfo.productAttr" :label="item.name" :key="item.id"
          align="center">
          <template #default="scope">
            {{ getProductSkuSp(scope.row, index) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('dialog.salePrice')" width="80" align="center">
          <template #default="scope">
            <el-input v-model="scope.row.price"></el-input>
          </template>
        </el-table-column>
        <el-table-column :label="t('dialog.productStock')" width="80" align="center">
          <template #default="scope">
            <el-input v-model="scope.row.stock"></el-input>
          </template>
        </el-table-column>
        <el-table-column :label="t('dialog.lowStockWarning')" width="100" align="center">
          <template #default="scope">
            <el-input v-model="scope.row.lowStock"></el-input>
          </template>
        </el-table-column>
      </el-table>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="editSkuInfo.dialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleEditSkuConfirm">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>
