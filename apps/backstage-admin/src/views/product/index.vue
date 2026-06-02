<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh, Upload, Warning } from '@element-plus/icons-vue'
import {
  getProductListAPI,
  productUpdateDeleteStatusAPI,
  productUpdateNewStatusAPI,
  productUpdateRecommendStatusAPI,
  productUpdatePublishStatusAPI
} from '@/apis/product'
import type { PmsProduct, ProductQueryParam } from '@/types/product'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

// State
const loading = ref(false)
const productList = ref<PmsProduct[]>([])
const total = ref(0)

// Query params
const queryParams = reactive<ProductQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  keyword: '',
  brandId: undefined,
  productCategoryId: undefined,
  publishStatus: undefined,
  verifyStatus: undefined
})

// Dialog state
const dialogVisible = ref(false)
const dialogTitle = ref('')
const currentProduct = ref<Partial<PmsProduct>>({})

// Options
const publishStatusOptions = [
  { value: 0, label: 'product.offline' },
  { value: 1, label: 'product.online' }
]

const verifyStatusOptions = [
  { value: 0, label: 'product.unaudited' },
  { value: 1, label: 'product.approved' }
]

const newStatusOptions = [
  { value: 0, label: 'product.notNew' },
  { value: 1, label: 'product.isNew' }
]

const recommandStatusOptions = [
  { value: 0, label: 'product.notRecommended' },
  { value: 1, label: 'product.recommended' }
]

// Methods
const getList = async () => {
  loading.value = true
  try {
    const res = await getProductListAPI(queryParams)
    productList.value = res.data.list
    total.value = res.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  queryParams.pageNum = 1
  getList()
}

const handleReset = () => {
  queryParams.keyword = ''
  queryParams.brandId = undefined
  queryParams.productCategoryId = undefined
  queryParams.publishStatus = undefined
  queryParams.verifyStatus = undefined
  queryParams.pageNum = 1
  getList()
}

const handleAdd = () => {
  dialogTitle.value = t('product.addProduct')
  currentProduct.value = {}
  dialogVisible.value = true
}

const handleEdit = (row: PmsProduct) => {
  dialogTitle.value = t('product.editProduct')
  currentProduct.value = { ...row }
  dialogVisible.value = true
}

const handleDelete = async (row: PmsProduct) => {
  try {
    await ElMessageBox.confirm(t('product.confirmDelete'), t('common.warning'), {
      type: 'warning'
    })
    await productUpdateDeleteStatusAPI({ ids: row.id!.toString(), deleteStatus: 1 })
    ElMessage.success(t('common.deleteSuccess'))
    getList()
  } catch {
    // cancelled
  }
}

const handleBatchDelete = async () => {
  const selected = productList.value.filter((p: PmsProduct) => (p as any).selected)
  if (selected.length === 0) {
    ElMessage.warning(t('product.selectProducts'))
    return
  }
  try {
    await ElMessageBox.confirm(t('product.confirmBatchDelete'), t('common.warning'), {
      type: 'warning'
    })
    await productUpdateDeleteStatusAPI({
      ids: selected.map(p => p.id!.toString()).join(','),
      deleteStatus: 1
    })
    ElMessage.success(t('common.deleteSuccess'))
    getList()
  } catch {
    // cancelled
  }
}

const handleUpdatePublishStatus = async (row: PmsProduct, status: number) => {
  try {
    await productUpdatePublishStatusAPI({ ids: row.id!.toString(), publishStatus: status })
    ElMessage.success(t('common.updateSuccess'))
    getList()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleUpdateNewStatus = async (row: PmsProduct, status: number) => {
  try {
    await productUpdateNewStatusAPI({ ids: row.id!.toString(), newStatus: status })
    ElMessage.success(t('common.updateSuccess'))
    getList()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleUpdateRecommendStatus = async (row: PmsProduct, status: number) => {
  try {
    await productUpdateRecommendStatusAPI({ ids: row.id!.toString(), recommendStatus: status })
    ElMessage.success(t('common.updateSuccess'))
    getList()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleSizeChange = (val: number) => {
  queryParams.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  queryParams.pageNum = val
  getList()
}

const getStatusLabel = (status: number | undefined, options: { value: number; label: string }[]) => {
  if (status === undefined || status === null) return '-'
  return options.find(o => o.value === status)?.label || '-'
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="product-container">
    <div class="toolbar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item :label="t('product.productName')">
          <el-input v-model="queryParams.keyword" :placeholder="t('product.productNamePlaceholder')" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item :label="t('product.publishStatus')">
          <el-select v-model="queryParams.publishStatus" :placeholder="t('common.select')" clearable>
            <el-option v-for="item in publishStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('product.verifyStatus')">
          <el-select v-model="queryParams.verifyStatus" :placeholder="t('common.select')" clearable>
            <el-option v-for="item in verifyStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
          <el-button type="primary" :icon="Plus" @click="handleAdd">{{ t('common.add') }}</el-button>
          <el-button type="danger" :icon="Delete" @click="handleBatchDelete">{{ t('common.batchDelete') }}</el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-table v-loading="loading" :data="productList" border stripe @selection-change="(rows: PmsProduct[]) => rows.forEach((p: PmsProduct) => (p as any).selected = true)">
      <el-table-column type="selection" width="55" />
      <el-table-column :label="t('product.id')" prop="id" width="80" />
      <el-table-column :label="t('product.productImage')" prop="pic" width="100">
        <template #default="{ row }">
          <el-image v-if="row.pic" :src="row.pic" fit="cover" style="width: 60px; height: 60px" />
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column :label="t('product.productName')" prop="name" min-width="200" show-overflow-tooltip />
      <el-table-column :label="t('product.productSn')" prop="productSn" width="140" />
      <el-table-column :label="t('product.price')" prop="price" width="100">
        <template #default="{ row }">
          ¥{{ row.price }}
        </template>
      </el-table-column>
      <el-table-column :label="t('product.stock')" prop="stock" width="80" />
      <el-table-column :label="t('product.sale')" prop="sale" width="80" />
      <el-table-column :label="t('product.publishStatus')" prop="publishStatus" width="100">
        <template #default="{ row }">
          <el-tag :type="row.publishStatus === 1 ? 'success' : 'info'">
            {{ getStatusLabel(row.publishStatus, publishStatusOptions) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('product.verifyStatus')" prop="verifyStatus" width="100">
        <template #default="{ row }">
          <el-tag :type="row.verifyStatus === 1 ? 'success' : 'warning'">
            {{ getStatusLabel(row.verifyStatus, verifyStatusOptions) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('product.newStatus')" prop="newStatus" width="90">
        <template #default="{ row }">
          <el-switch
            :model-value="row.newStatus === 1"
            @change="(val: boolean | string | number) => handleUpdateNewStatus(row, !!val ? 1 : 0)"
          />
        </template>
      </el-table-column>
      <el-table-column :label="t('product.recommandStatus')" prop="recommandStatus" width="100">
        <template #default="{ row }">
          <el-switch
            :model-value="row.recommandStatus === 1"
            @change="(val: boolean | string | number) => handleUpdateRecommendStatus(row, !!val ? 1 : 0)"
          />
        </template>
      </el-table-column>
      <el-table-column :label="t('common.operation')" width="200" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link :icon="Edit" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
          <el-button type="primary" link :icon="Upload" @click="handleUpdatePublishStatus(row, row.publishStatus === 1 ? 0 : 1)">
            {{ row.publishStatus === 1 ? t('product.offline') : t('product.online') }}
          </el-button>
          <el-button type="danger" link :icon="Delete" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="pagination">
      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :page-sizes="PAGE_SIZE_OPTIONS"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>

    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px" destroy-on-close>
      <el-form :model="currentProduct" label-width="100px">
        <el-form-item :label="t('product.productName')">
          <el-input v-model="currentProduct.name" />
        </el-form-item>
        <el-form-item :label="t('product.productSn')">
          <el-input v-model="currentProduct.productSn" />
        </el-form-item>
        <el-form-item :label="t('product.price')">
          <el-input-number v-model="currentProduct.price" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('product.stock')">
          <el-input-number v-model="currentProduct.stock" :min="0" />
        </el-form-item>
        <el-form-item :label="t('product.unit')">
          <el-input v-model="currentProduct.unit" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="dialogVisible = false">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.product-container {
  padding: 20px;
  
  .toolbar {
    margin-bottom: 20px;
  }
  
  .pagination {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
  }
  
  :deep(.el-table) {
    .el-table__row {
      td:last-child {
        .el-button + .el-button {
          margin-left: 8px;
        }
      }
    }
  }
}
</style>