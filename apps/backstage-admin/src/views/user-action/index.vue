<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, Delete, View } from '@element-plus/icons-vue'
import type { TabPaneName } from 'element-plus'
import {
  getCollectionListAPI,
  getHistoryListAPI,
  getAttentionListAPI,
  getCommentListAPI,
  getUserActionStatisticsAPI,
  deleteCollectionAPI,
  deleteHistoryAPI,
  deleteAttentionAPI,
  updateCommentStatusAPI,
  deleteCommentAPI,
  batchDeleteCollectionsAPI,
  batchDeleteHistoriesAPI,
  batchDeleteAttentionsAPI,
  batchDeleteCommentsAPI,
  type ProductCollection,
  type ReadHistory,
  type BrandAttention,
  type ProductComment,
  type UserActionQueryParam,
  type UserActionStatistics
} from '@/apis/userAction'
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'

const loading = ref(false)
const activeTab = ref('collection')
const statistics = ref<UserActionStatistics>({
  totalCollections: 0,
  totalHistories: 0,
  totalAttentions: 0,
  totalComments: 0,
  visibleComments: 0
})

const query = reactive<UserActionQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  userId: undefined,
  productId: undefined,
  productName: undefined,
  brandId: undefined,
  brandName: undefined,
  showStatus: undefined,
  star: undefined
})

const collectionList = ref<ProductCollection[]>([])
const historyList = ref<ReadHistory[]>([])
const attentionList = ref<BrandAttention[]>([])
const commentList = ref<ProductComment[]>([])
const total = ref(0)

const showStatusOptions = [
  { label: 'Hidden', value: 0 },
  { label: 'Visible', value: 1 }
]

const starOptions = [
  { label: '1 Star', value: 1 },
  { label: '2 Stars', value: 2 },
  { label: '3 Stars', value: 3 },
  { label: '4 Stars', value: 4 },
  { label: '5 Stars', value: 5 }
]

const getStarTagType = (star: number): 'info' | 'warning' | 'danger' | 'success' | 'primary' => {
  const map: Record<number, 'info' | 'warning' | 'danger' | 'success' | 'primary'> = {
    1: 'info',
    2: 'warning',
    3: 'primary',
    4: 'success',
    5: 'danger'
  }
  return map[star] || 'info'
}

const getShowStatusTagType = (status: number): 'success' | 'info' => {
  return status === 1 ? 'success' : 'info'
}

const getShowStatusText = (status: number): string => {
  return status === 1 ? 'Visible' : 'Hidden'
}

const getList = async () => {
  loading.value = true
  try {
    let res
    switch (activeTab.value) {
      case 'collection':
        res = await getCollectionListAPI(query)
        if (res.data) {
          collectionList.value = res.data.list || []
          total.value = res.data.total || 0
        }
        break
      case 'history':
        res = await getHistoryListAPI(query)
        if (res.data) {
          historyList.value = res.data.list || []
          total.value = res.data.total || 0
        }
        break
      case 'attention':
        res = await getAttentionListAPI(query)
        if (res.data) {
          attentionList.value = res.data.list || []
          total.value = res.data.total || 0
        }
        break
      case 'comment':
        res = await getCommentListAPI(query)
        if (res.data) {
          commentList.value = res.data.list || []
          total.value = res.data.total || 0
        }
        break
    }
  } catch (error) {
    console.error('Failed to load list:', error)
  } finally {
    loading.value = false
  }
}

const getStatistics = async () => {
  try {
    const res = await getUserActionStatisticsAPI()
    if (res.data) {
      statistics.value = res.data
    }
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

const handleSearch = () => {
  query.pageNum = 1
  getList()
}

const handleReset = () => {
  query.pageNum = 1
  query.userId = undefined
  query.productId = undefined
  query.productName = undefined
  query.brandId = undefined
  query.brandName = undefined
  query.showStatus = undefined
  query.star = undefined
  getList()
}

const handlePageChange = (page: number) => {
  query.pageNum = page
  getList()
}

const handleSizeChange = (size: number) => {
  query.pageSize = size
  query.pageNum = 1
  getList()
}

const handleRefresh = () => {
  getList()
  getStatistics()
}

const handleTabChange = (name: TabPaneName) => {
  activeTab.value = name as string
  query.pageNum = 1
  getList()
}

const handleDelete = async (row: any, type: string) => {
  try {
    await ElMessageBox.confirm(
      `Are you sure you want to delete this item?`,
      'Confirm Delete',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    switch (type) {
      case 'collection':
        await deleteCollectionAPI(row.id)
        break
      case 'history':
        await deleteHistoryAPI(row.id)
        break
      case 'attention':
        await deleteAttentionAPI(row.id)
        break
      case 'comment':
        await deleteCommentAPI(row.id)
        break
    }
    ElMessage.success('Deleted successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete:', error)
    }
  }
}

const handleUpdateStatus = async (row: ProductComment, status: number) => {
  try {
    await updateCommentStatusAPI(row.id, status)
    ElMessage.success('Status updated successfully')
    getList()
    getStatistics()
  } catch (error) {
    console.error('Failed to update status:', error)
  }
}

onMounted(() => {
  getList()
  getStatistics()
})
</script>

<template>
  <div class="user-action-container">
    <el-card class="statistics-card">
      <el-row :gutter="20">
        <el-col :span="4">
          <div class="stat-item">
            <div class="stat-value">{{ statistics.totalCollections }}</div>
            <div class="stat-label">Collections</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-histories">
            <div class="stat-value">{{ statistics.totalHistories }}</div>
            <div class="stat-label">Histories</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-attentions">
            <div class="stat-value">{{ statistics.totalAttentions }}</div>
            <div class="stat-label">Brand attentions</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-comments">
            <div class="stat-value">{{ statistics.totalComments }}</div>
            <div class="stat-label">Total Comments</div>
          </div>
        </el-col>
        <el-col :span="4">
          <div class="stat-item stat-visible">
            <div class="stat-value">{{ statistics.visibleComments }}</div>
            <div class="stat-label">Visible Comments</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="filter-card">
      <el-form :inline="true" :model="query" class="filter-form">
        <el-form-item label="User ID">
          <el-input v-model="query.userId" placeholder="User ID" clearable style="width: 120px" type="number" />
        </el-form-item>
        <el-form-item v-if="activeTab === 'collection' || activeTab === 'history' || activeTab === 'comment'" label="Product Name">
          <el-input v-model="query.productName" placeholder="Product Name" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item v-if="activeTab === 'attention'" label="Brand Name">
          <el-input v-model="query.brandName" placeholder="Brand Name" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item v-if="activeTab === 'comment'" label="Status">
          <el-select v-model="query.showStatus" placeholder="Select Status" clearable style="width: 120px">
            <el-option v-for="item in showStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item v-if="activeTab === 'comment'" label="Star">
          <el-select v-model="query.star" placeholder="Select Star" clearable style="width: 120px">
            <el-option v-for="item in starOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
          <el-button :icon="Refresh" @click="handleRefresh">Refresh</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-tabs v-model="activeTab" @tab-change="handleTabChange">
        <el-tab-pane label="Collections" name="collection">
          <el-table v-loading="loading" :data="collectionList" border style="width: 100%">
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="productId" label="Product ID" width="100" />
            <el-table-column prop="productName" label="Product Name" min-width="150" />
            <el-table-column prop="productPic" label="Product Pic" width="100">
              <template #default="{ row }">
                <el-image v-if="row.productPic" :src="row.productPic" style="width: 50px; height: 50px" fit="cover" />
              </template>
            </el-table-column>
            <el-table-column prop="createTime" label="Create Time" width="170" />
            <el-table-column label="Actions" width="100" fixed="right">
              <template #default="{ row }">
                <el-button type="danger" link @click="handleDelete(row, 'collection')">Delete</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Histories" name="history">
          <el-table v-loading="loading" :data="historyList" border style="width: 100%">
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="productId" label="Product ID" width="100" />
            <el-table-column prop="productName" label="Product Name" min-width="150" />
            <el-table-column prop="productPic" label="Product Pic" width="100">
              <template #default="{ row }">
                <el-image v-if="row.productPic" :src="row.productPic" style="width: 50px; height: 50px" fit="cover" />
              </template>
            </el-table-column>
            <el-table-column prop="createTime" label="Create Time" width="170" />
            <el-table-column label="Actions" width="100" fixed="right">
              <template #default="{ row }">
                <el-button type="danger" link @click="handleDelete(row, 'history')">Delete</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Brand Attentions" name="attention">
          <el-table v-loading="loading" :data="attentionList" border style="width: 100%">
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="brandId" label="Brand ID" width="100" />
            <el-table-column prop="brandName" label="Brand Name" min-width="150" />
            <el-table-column prop="brandLogo" label="Brand Logo" width="100">
              <template #default="{ row }">
                <el-image v-if="row.brandLogo" :src="row.brandLogo" style="width: 50px; height: 50px" fit="cover" />
              </template>
            </el-table-column>
            <el-table-column prop="createTime" label="Create Time" width="170" />
            <el-table-column label="Actions" width="100" fixed="right">
              <template #default="{ row }">
                <el-button type="danger" link @click="handleDelete(row, 'attention')">Delete</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Comments" name="comment">
          <el-table v-loading="loading" :data="commentList" border style="width: 100%">
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="memberNickName" label="Nickname" width="120" />
            <el-table-column prop="productName" label="Product Name" min-width="120" />
            <el-table-column prop="star" label="Star" width="80">
              <template #default="{ row }">
                <el-tag :type="getStarTagType(row.star)">{{ row.star }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="content" label="Content" min-width="150" show-overflow-tooltip />
            <el-table-column prop="showStatus" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getShowStatusTagType(row.showStatus)">{{ getShowStatusText(row.showStatus) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="createTime" label="Create Time" width="170" />
            <el-table-column label="Actions" width="150" fixed="right">
              <template #default="{ row }">
                <el-button v-if="row.showStatus === 0" type="success" link @click="handleUpdateStatus(row, 1)">Show</el-button>
                <el-button v-if="row.showStatus === 1" type="info" link @click="handleUpdateStatus(row, 0)">Hide</el-button>
                <el-button type="danger" link @click="handleDelete(row, 'comment')">Delete</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="query.pageNum"
          v-model:page-size="query.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.user-action-container {
  padding: 20px;
}

.statistics-card {
  margin-bottom: 20px;
}

.stat-item {
  text-align: center;
  padding: 10px;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.stat-histories .stat-value {
  color: #409eff;
}

.stat-attentions .stat-value {
  color: #e6a23c;
}

.stat-comments .stat-value {
  color: #f56c6c;
}

.stat-visible .stat-value {
  color: #67c23a;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  margin-bottom: 0;
}

.table-card {
  min-height: 400px;
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>