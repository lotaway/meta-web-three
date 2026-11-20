<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type Action } from 'element-plus'
import { getBrandListAPI, brandUpdateShowStatusAPI, brandUpdateFactoryStatusAPI, brandDeleteByIdAPI } from '@/apis/brand'
import { Search, Tickets } from '@element-plus/icons-vue'
import type { PmsBrand } from '@/types/brand'
import type { PageParam } from '@/types/common'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const router = useRouter()

// Brand list query params
const listQuery = ref<PageParam>({
  keyword: '',
  pageNum: 1,
  pageSize: 10
})
// Brand list data
const list = ref<PmsBrand[]>([])
// Selected rows in the table
const multipleSelection = ref<PmsBrand[]>([])
// Table data loading progress bar
const listLoading = ref(true)
// Pagination component params
const total = ref(0)
// Get brand list data
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getBrandListAPI(listQuery.value)
    listLoading.value = false
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load brand list:', error)
  }
}

// Initialize list info after mount
onMounted(() => {
  getList()
})

// Handle brand search
const handleSearchBrand = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Handle add brand
const handleAddBrand = () => {
  router.push({ path: '/pms/addBrand' })
}

// Handle edit brand operation
const handleUpdateBrand = (index: number, row: PmsBrand) => {
  router.push({ path: '/pms/updateBrand', query: { id: row.id } })
}

// Handle delete brand operation
const handleDeleteBrand = async (index: number, row: PmsBrand) => {
  ElMessageBox.confirm(t('dialog.confirmDeleteBrand'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
    callback: async (action: Action) => {
      if (action === 'confirm') {
        await brandDeleteByIdAPI(row.id!)
        ElMessage.success(t('common.deleteSuccess'))
        getList()
      }
    },
  })
}

// Handle table selection state change
const handleSelectionChange = (val: PmsBrand[]) => {
  multipleSelection.value = val
}

// Get product list
const getProductList = (index: number, row: PmsBrand) => {
  console.log(index, row)
}

// Get product review list
const getProductCommentList = (index: number, row: PmsBrand) => {
  console.log(index, row)
}

// Handle manufacturer status change
const handleFactoryStatusChange = async (index: number, row: PmsBrand) => {

  try {
    await brandUpdateFactoryStatusAPI({
      ids: row.id!.toString(),
      factoryStatus: row.factoryStatus
    })
    ElMessage.success(t('message.modifySuccess'))
  } catch (error) {
    // Roll back state if the update fails
    row.factoryStatus = row.factoryStatus === 0 ? 1 : 0
    console.error('Failed to update manufacturer status:', error)
  }
}

// Handle display status change
const handleShowStatusChange = async (index: number, row: PmsBrand) => {
  try {
    await brandUpdateShowStatusAPI({
      ids: row.id!.toString(),
      showStatus: row.showStatus
    })
    ElMessage.success(t('message.modifySuccess'))
  } catch (error) {
    // Roll back state if the update fails
    row.showStatus = row.showStatus === 0 ? 1 : 0
    console.error('Failed to update display status:', error)
  }
}

// Handle page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Handle current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Batch operation type
const operates = ref([
  {
    label: "显示品牌",
    value: "showBrand"
  },
  {
    label: "隐藏品牌",
    value: "hideBrand"
  }
])

// Bottom multi-select batch operation type
const operateType = ref<string | null>()

// Handle batch operations
const handleBatchOperate = async () => {
  if (!multipleSelection.value || multipleSelection.value.length < 1) {
    ElMessage.warning(t('message.selectOneRecord'))
    return
  }
  let showStatus = 0
  if (operateType.value === 'showBrand') {
    showStatus = 1
  } else if (operateType.value === 'hideBrand') {
    showStatus = 0
  } else {
    ElMessage.warning(t('message.selectOperationType'))
    return
  }
  const idsArr = multipleSelection.value.map(item => item.id)
  try {
    await brandUpdateShowStatusAPI({
      ids: idsArr.join(','),
      showStatus: showStatus
    })
    getList()
    ElMessage.success(t('message.modifySuccess'))
  } catch (error) {
    console.error('Failed to perform batch operation:', error)
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
        <span>筛选搜索</span>
        <el-button style="float: right" @click="handleSearchBrand()" type="primary">
          查询结果
        </el-button>
      </div>
      <div style="margin-top: 15px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item label="输入搜索：">
            <el-input style="width: 203px" v-model="listQuery.keyword" placeholder="品牌名称/关键字"></el-input>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>数据列表</span>
      <el-button class="btn-add" @click="handleAddBrand()">
        添加
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="brandTable" :data="list" style="width: 100%" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="编号" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="品牌名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="品牌首字母" width="100" align="center">
          <template #default="scope">{{ scope.row.firstLetter }}</template>
        </el-table-column>
        <el-table-column label="排序" width="100" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="品牌制造商" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleFactoryStatusChange(scope.$index, scope.row as PmsBrand)" :active-value="1"
              :inactive-value="0" v-model="scope.row.factoryStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="是否显示" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleShowStatusChange(scope.$index, scope.row as PmsBrand)" :active-value="1" :inactive-value="0"
              v-model="scope.row.showStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="相关" width="220" align="center">
          <template #default="scope">
            <span>商品：</span>
            <el-button size="small" type="primary" link @click="getProductList(scope.$index, scope.row as PmsBrand)">100
            </el-button>
            <span>评价：</span>
            <el-button size="small" type="primary" link @click="getProductCommentList(scope.$index, scope.row as PmsBrand)">1000
            </el-button>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleUpdateBrand(scope.$index, scope.row as PmsBrand)">编辑
            </el-button>
            <el-button size="small" type="danger" @click="handleDeleteBrand(scope.$index, scope.row as PmsBrand)">删除
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
  </div>
</template>

<style rel="stylesheet/scss" lang="scss" scoped></style>
