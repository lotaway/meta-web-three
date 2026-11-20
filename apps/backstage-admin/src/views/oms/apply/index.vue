<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/datetime'
import { getReturnApplyListAPI, returnApplyDeleteByIdsAPI } from '@/apis/returnApply'
import type { OmsOrderReturnApply, ReturnApplyQueryParam } from '@/types/returnApply'

// Get route object
const router = useRouter()

// Default processing status
const defaultStatusOptions = [
  {
    label: '待处理',
    value: 0
  },
  {
    label: '退货中',
    value: 1
  },
  {
    label: '已完成',
    value: 2
  },
  {
    label: '已拒绝',
    value: 3
  }
]

// List query params
const listQuery = ref<ReturnApplyQueryParam>({
  pageNum: 1,
  pageSize: 10
})
// Status options
const statusOptions = ref(Object.assign({}, defaultStatusOptions))
// List data
const list = ref<OmsOrderReturnApply[]>([])
// Total
const total = ref(0)
// Loading state
const listLoading = ref(false)
// Multi-select data
const multipleSelection = ref<OmsOrderReturnApply[]>([])
// Get list
const getList = async () => {
  listLoading.value = true
  const res = await getReturnApplyListAPI(listQuery.value)
  listLoading.value = false
  list.value = res.data.list
  total.value = res.data.total
}

// Get list after mount
onMounted(() => {
  getList()
})

// Operation type
const operateType = ref()
// Operation options
const operateOptions = ref([
  {
    label: "批量删除",
    value: 1
  }
])

// Format status
const formatStatus = (status: number) => {
  return defaultStatusOptions.find(item => item.value === status)?.label
}

// Format refund amount
const formatReturnAmount = (row: OmsOrderReturnApply) => {
  return row.productRealPrice * row.productCount
}

// Handle selection change
const handleSelectionChange = (val: OmsOrderReturnApply[]) => {
  multipleSelection.value = val
}

// Reset search
const handleResetSearch = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
}

// Search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// View details
const handleViewDetail = (index: number, row: OmsOrderReturnApply) => {
  router.push({ path: '/oms/returnApplyDetail', query: { id: row.id } })
}

// Batch operations
const handleBatchOperate = async () => {
  if (!multipleSelection.value || multipleSelection.value.length < 1) {
    ElMessage({
      message: '请选择要操作的申请',
      type: 'warning',
      duration: 1000
    })
    return
  }
  if (operateType.value === 1) {
    // Batch delete
    await ElMessageBox.confirm('是否要进行删除操作?', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await returnApplyDeleteByIdsAPI({ ids: multipleSelection.value.map(item => item.id).join(',') })
    getList()
    ElMessage({
      type: 'success',
      message: '删除成功!'
    })
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


</script>

<template>
  <div class="app-container">
    <el-card class="filter-container" shadow="never">
      <div>
        <el-icon class="el-icon-middle">
          <Search />
        </el-icon>
        <span>筛选搜索</span>
        <el-button style="float:right" type="primary" @click="handleSearchList()">
          查询搜索
        </el-button>
        <el-button style="float:right;margin-right: 15px" @click="handleResetSearch()">
          重置
        </el-button>
      </div>
      <div style="margin-top: 20px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item label="输入搜索：">
            <el-input v-model="listQuery.id" class="input-width" placeholder="服务单号"></el-input>
          </el-form-item>
          <el-form-item label="处理状态：">
            <el-select v-model="listQuery.status" placeholder="全部" clearable class="input-width">
              <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="申请时间：">
            <el-date-picker class="input-width" v-model="listQuery.createTime" value-format="yyyy-MM-dd" type="date"
              placeholder="请选择时间">
            </el-date-picker>
          </el-form-item>
          <el-form-item label="操作人员：">
            <el-input v-model="listQuery.handleMan" class="input-width" placeholder="全部"></el-input>
          </el-form-item>
          <el-form-item label="处理时间：">
            <el-date-picker class="input-width" v-model="listQuery.handleTime" value-format="yyyy-MM-dd" type="date"
              placeholder="请选择时间">
            </el-date-picker>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>数据列表</span>
    </el-card>
    <div class="table-container">
      <el-table ref="returnApplyTable" :data="list" style="width: 100%;" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="服务单号" width="180" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="申请时间" width="180" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.createTime) }}</template>
        </el-table-column>
        <el-table-column label="用户账号" align="center">
          <template #default="scope">{{ scope.row.memberUsername }}</template>
        </el-table-column>
        <el-table-column label="退款金额" width="180" align="center">
          <template #default="scope">￥{{ formatReturnAmount(scope.row as OmsOrderReturnApply) }}</template>
        </el-table-column>
        <el-table-column label="申请状态" width="180" align="center">
          <template #default="scope">{{ formatStatus(scope.row.status) }}</template>
        </el-table-column>
        <el-table-column label="处理时间" width="180" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.handleTime) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleViewDetail(scope.$index, scope.row as OmsOrderReturnApply)">查看详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" placeholder="批量操作">
        <el-option v-for="item in operateOptions" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        确定
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" v-model:current-page="listQuery.pageNum"
        :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="total">
      </el-pagination>
    </div>
  </div>
</template>

<style scoped>
.input-width {
  width: 203px
}
</style>
