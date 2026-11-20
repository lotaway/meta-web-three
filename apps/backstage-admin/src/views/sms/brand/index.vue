<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Search, Tickets } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getHomeBrandListAPI, homeBrandUpdateRecommendStatusAPI, homeBrandDeleteByIdsAPI, homeBrandCreateAPI, homeBrandUpdateSortAPI } from '@/apis/homeBrand'
import { getBrandListAPI } from '@/apis/brand'
import type { HomeBrandQueryParam, SmsHomeBrand } from '@/types/homeBrand'
import type { PmsBrand } from '@/types/brand'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// List query params
const listQuery = ref<HomeBrandQueryParam>({
  pageNum: 1,
  pageSize: 10,
})
// Home brand list data
const list = ref<SmsHomeBrand[]>([])
// Total count
const total = ref<number>(0)
// Loading state
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getHomeBrandListAPI(listQuery.value)
    listLoading.value = false
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load list:', error)
  }
}

// Fetch data after the page loads
onMounted(() => {
  getList()
})

// Selected rows in the table
const multipleSelection = ref<SmsHomeBrand[]>([])

const recommendOptions = ref([
  {
    label: '未推荐',
    value: 0
  },
  {
    label: '推荐中',
    value: 1
  }
])

const operates = ref([
  {
    label: "设为推荐",
    value: 0
  },
  {
    label: "取消推荐",
    value: 1
  },
  {
    label: "删除",
    value: 2
  }
])
const operateType = ref<number>()

// Select-brand dialog visibility
const selectDialogVisible = ref(false)
// Data in the select-brand dialog
const dialogData = ref({
  list: [] as PmsBrand[],
  total: 0,
  multipleSelection: [] as PmsBrand[],
  listQuery: {
    keyword: '',
    showStatus: 1,
    pageNum: 1,
    pageSize: 5
  }
})
// Get select-brand dialog list
const getDialogList = async () => {
  try {
    const res = await getBrandListAPI(dialogData.value.listQuery)
    dialogData.value.list = res.data.list
    dialogData.value.total = res.data.total
  } catch (error) {
    console.error('Failed to load dialog list:', error)
  }
}

// Set sort dialog visibility
const sortDialogVisible = ref(false)
// Set data in the sort dialog
const sortDialogData = ref({ sort: 0, id: 0 })

// Reset search
const handleResetSearch = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10,
  }
}

// Search list
const handleSearchList = async () => {
  listQuery.value.pageNum = 1
  await getList()
}

// Handle table selection change
const handleSelectionChange = (val: SmsHomeBrand[]) => {
  multipleSelection.value = val
}

// Page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Handle recommendation status change
const handleRecommendStatusStatusChange = async (index: number, row: SmsHomeBrand) => {
  await updateRecommendStatusStatus([row.id!], row.recommendStatus!)
}

// Handle delete
const handleDelete = async (index: number, row: SmsHomeBrand) => {
  await deleteBrand([row.id!])
}

// Handle batch operations
const handleBatchOperate = async () => {
  if (multipleSelection.value.length < 1) {
    ElMessage.warning({
      message: '请选择一条记录',
      duration: 1000
    })
    return
  }
  const ids = multipleSelection.value.map(item => item.id!)
  if (operateType.value === 0) {
    // Set as recommended
    await updateRecommendStatusStatus(ids, 1)
  } else if (operateType.value === 1) {
    // Unset recommendation
    await updateRecommendStatusStatus(ids, 0)
  } else if (operateType.value === 2) {
    // Delete
    await deleteBrand(ids)
  } else {
    ElMessage.warning({
      message: '请选择批量操作类型',
      duration: 1000
    })
    return
  }
}

// Handle brand selection
const handleSelectBrand = () => {
  selectDialogVisible.value = true
  dialogData.value.listQuery.keyword = ''
  getDialogList()
}

// Handle dialog search
const handleSelectSearch = () => {
  getDialogList()
}

// Dialog page size change
const handleDialogSizeChange = (val: number) => {
  dialogData.value.listQuery.pageNum = 1
  dialogData.value.listQuery.pageSize = val
  getDialogList()
}

// Dialog current page change
const handleDialogCurrentChange = (val: number) => {
  dialogData.value.listQuery.pageNum = val
  getDialogList()
}

// Dialog selection change
const handleDialogSelectionChange = (val: PmsBrand[]) => {
  dialogData.value.multipleSelection = val
}

// Confirm selection dialog
const handleSelectDialogConfirm = async () => {
  if (dialogData.value.multipleSelection.length < 1) {
    ElMessage.warning({
      message: '请选择一条记录',
      duration: 1000
    })
    return
  }
  try {
    await ElMessageBox.confirm(t('dialog.confirmAdd'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    const homeBrandList: SmsHomeBrand[] = dialogData.value.multipleSelection.map(item => ({
      brandId: item.id!,
      brandName: item.name
    }))
    await homeBrandCreateAPI(homeBrandList)
    selectDialogVisible.value = false
    dialogData.value.multipleSelection = []
    await getList()
    ElMessage.success({
      type: 'success',
      message: '添加成功!'
    })
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to confirm operation:', error)
    }
  }
}

// Edit sort
const handleEditSort = (index: number, row: SmsHomeBrand) => {
  sortDialogVisible.value = true
  sortDialogData.value.sort = row.sort!
  sortDialogData.value.id = row.id!
}

// Update sort
const handleUpdateSort = async () => {
  if (!sortDialogData.value.id) return
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeSort'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeBrandUpdateSortAPI(sortDialogData.value)
    sortDialogVisible.value = false
    await getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to update sort order:', error)
    }
  }
}

// Update recommendation status
const updateRecommendStatusStatus = async (ids: number[], status: number) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeRecommend'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeBrandUpdateRecommendStatusAPI({
      ids: ids.join(','),
      recommendStatus: status
    })
    await getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch (error) {
    console.error('Failed to update:', error)
    await getList()
  }
}

// Delete brand
const deleteBrand = async (ids: number[]) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteRecommend'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeBrandDeleteByIdsAPI({ ids: ids.join(',') })
    await getList()
    ElMessage.success({
      type: 'success',
      message: '删除成功!'
    })
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete operation:', error)
    }
  }
}

// Recommendation status filter
const formatRecommendStatus = (status: number) => {
  if (status === 1) {
    return '推荐中'
  } else {
    return '未推荐'
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
        <span>{{ t('role.page.filterSearch') }}</span>
        <el-button style="float:right" type="primary" @click="handleSearchList()">
          {{ t('role.page.querySearch') }}
        </el-button>
        <el-button style="float:right;margin-right: 15px" @click="handleResetSearch()">
          {{ t('common.reset') }}
        </el-button>
      </div>
      <div style="margin-top: 15px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item :label="t('product.brandName')">
            <el-input v-model="listQuery.brandName" :placeholder="t('promotion.brandName')"></el-input>
          </el-form-item>
          <el-form-item :label="t('promotion.recommendStatus')">
            <el-select v-model="listQuery.recommendStatus" :placeholder="t('product.all')" clearable style="width: 150px;">
              <el-option v-for="item in recommendOptions" :key="item.value" :label="item.label" :value="item.value">
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
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="handleSelectBrand()">{{ t('promotion.selectBrand') }}</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="homeBrandTable" :data="list" style="width: 100%;" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="120" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="品牌名称" align="center">
          <template #default="scope">{{ scope.row.brandName }}</template>
        </el-table-column>
        <el-table-column label="是否推荐" width="200" align="center">
          <template #default="scope">
            <el-switch @change="handleRecommendStatusStatusChange(scope.$index, scope.row as SmsHomeBrand)" :active-value="1"
              :inactive-value="0" v-model="scope.row.recommendStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="排序" width="160" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="状态" width="160" align="center">
          <template #default="scope">{{ formatRecommendStatus(scope.row.recommendStatus) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleEditSort(scope.$index, scope.row as SmsHomeBrand)">{{ t('promotion.setSort') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as SmsHomeBrand)">{{ t('common.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" placeholder="批量操作">
        <el-option v-for="item in operates" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        确定
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
    <el-dialog title="选择品牌" v-model="selectDialogVisible" width="40%">
      <el-input v-model="dialogData.listQuery.keyword" style="width: 250px;margin-bottom: 20px" placeholder="品牌名称搜索">
        <template #append>
          <el-button :icon="Search" @click="handleSelectSearch()"></el-button>
        </template>
      </el-input>
      <el-table :data="dialogData.list" @selection-change="handleDialogSelectionChange" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="品牌名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="相关" width="220" align="center">
          <template #default="scope">
            {{ t('product.product') }}<span class="color-main">{{ scope.row.productCount }}</span>
            {{ t('product.review') }}<span class="color-main">{{ scope.row.productCommentCount }}</span>
          </template>
        </el-table-column>
      </el-table>
      <div class="pagination-container">
        <el-pagination background @size-change="handleDialogSizeChange" @current-change="handleDialogCurrentChange"
          layout="prev, pager, next" v-model:current-page="dialogData.listQuery.pageNum"
          :page-size="dialogData.listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="dialogData.total">
        </el-pagination>
      </div>
      <div style="clear: both;"></div>
      <template #footer>
        <div>
          <el-button size="small" @click="selectDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button size="small" type="primary" @click="handleSelectDialogConfirm()">{{ t('common.confirm') }}</el-button>
        </div>
      </template>
    </el-dialog>
    <el-dialog title="设置排序" v-model="sortDialogVisible" width="40%">
      <el-form :model="sortDialogData" label-width="150px">
        <el-form-item label="排序：">
          <el-input v-model="sortDialogData.sort" style="width: 200px"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span>
          <el-button @click="sortDialogVisible = false" size="small">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleUpdateSort" size="small">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped></style>
