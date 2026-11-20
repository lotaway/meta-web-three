<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getHomeAdvertiseListAPI, homeAdvertiseUpdateStatusAPI, deleteHomeAdvertiseAPI } from '@/apis/homeAdvertise'
import { formatDateTime } from '@/utils/datetime'
import type { HomeAdvertiseQueryParam, SmsHomeAdvertise } from '@/types/homeAdvertist'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route
const router = useRouter()

// List query params
const listQuery = ref<HomeAdvertiseQueryParam>({
  pageNum: 1,
  pageSize: 10
})
// Carousel position options
const typeOptions = ref([
  {
    label: 'PC首页轮播',
    value: 0
  },
  {
    label: 'APP首页轮播',
    value: 1
  }
])
// List data
const list = ref<SmsHomeAdvertise[]>([])
// Total count
const total = ref(0)
// Loading state
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true
  const res = await getHomeAdvertiseListAPI(listQuery.value)
  listLoading.value = false
  list.value = res.data.list
  total.value = res.data.total
}

// Run after component mounts
onMounted(() => {
  getList()
})

// Selected items
const multipleSelection = ref<SmsHomeAdvertise[]>([])
// Batch operation type
const operates = ref([
  {
    label: "删除",
    value: 0
  }
])
// Session batch operation type
const operateType = ref()

// Reset search
const handleResetSearch = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
}

// Search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Handle table selection change
const handleSelectionChange = (val: SmsHomeAdvertise[]) => {
  multipleSelection.value = val
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

// Handle status update
const handleUpdateStatus = async (index: number, row: SmsHomeAdvertise) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeOnlineStatus'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeAdvertiseUpdateStatusAPI({ id: row.id!, status: row.status! })
    getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch {
    getList()
  }
}

// Delete ad
const deleteHomeAdvertise = async (ids: number[]) => {
  await ElMessageBox.confirm(t('dialog.confirmDeleteAd'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await deleteHomeAdvertiseAPI({ ids: ids.join(',') })
  getList()
  ElMessage.success({
    type: 'success',
    message: '删除成功!'
  })
}

// Handle delete
const handleDelete = async (index: number, row: SmsHomeAdvertise) => {
  await deleteHomeAdvertise([row.id!])
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
  if (operateType.value === 0) {
    // Delete
    await deleteHomeAdvertise(multipleSelection.value.map(item => item.id!))
  } else {
    ElMessage.warning({
      message: '请选择批量操作类型',
      duration: 1000
    })
  }
}

// Handle add
const handleAdd = () => {
  router.push({ path: '/sms/addAdvertise' })
}

// Handle update
const handleUpdate = (index: number, row: SmsHomeAdvertise) => {
  router.push({ path: '/sms/updateAdvertise', query: { id: row.id } })
}

// Format type
const formatType = (type: number) => {
  if (type === 1) {
    return 'APP首页轮播'
  } else {
    return 'PC首页轮播'
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
      <div style="margin-top: 25px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item :label="t('promotion.adName')">
            <el-input v-model="listQuery.name" class="input-width" :placeholder="t('promotion.adNamePlain')"></el-input>
          </el-form-item>
          <el-form-item :label="t('promotion.adPosition')">
            <el-select v-model="listQuery.type" :placeholder="t('product.all')" clearable class="input-width">
              <el-option v-for="item in typeOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item :label="t('promotion.expireTime')">
            <el-date-picker class="input-width" v-model="listQuery.endTime" value-format="yyyy-MM-dd" type="date"
              :placeholder="t('oms.order.list.createTimePlaceholder')">
            </el-date-picker>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="handleAdd()">{{ t('promotion.addAd') }}</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="homeAdvertiseTable" :data="list" style="width: 100%" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="广告名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="广告位置" width="120" align="center">
          <template #default="scope">{{ formatType(scope.row.type) }}</template>
        </el-table-column>
        <el-table-column label="广告图片" width="180" align="center">
          <template #default="scope"><img style="height: 80px;width: 150px;" :src="scope.row.pic"></template>
        </el-table-column>
        <el-table-column label="时间" width="240" align="center">
          <template #default="scope">
            <p>{{ t('mes.productionTask.startTime') }}{{ formatDateTime(scope.row.startTime) }}</p>
            <p>{{ t('promotion.expireTime') }}{{ formatDateTime(scope.row.endTime) }}</p>
          </template>
        </el-table-column>
        <el-table-column label="上线/下线" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleUpdateStatus(scope.$index, scope.row)" :active-value="1" :inactive-value="0"
              v-model="scope.row.status">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="点击次数" width="100" align="center">
          <template #default="scope">{{ scope.row.clickCount }}</template>
        </el-table-column>
        <el-table-column label="生成订单" width="100" align="center">
          <template #default="scope">{{ scope.row.orderCount }}</template>
        </el-table-column>
        <el-table-column label="操作" width="120" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleUpdate(scope.$index, scope.row)">{{ t('common.edit') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row)">{{ t('common.delete') }}
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

<style scoped>
.input-width {
  width: 203px;
}
</style>
