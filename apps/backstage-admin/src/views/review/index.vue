<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, View, Check, Close, ChatLineRound } from '@element-plus/icons-vue'
import {
  getReviewListAPI,
  getReviewByIdAPI,
  approveReviewAPI,
  rejectReviewAPI,
  replyReviewAPI,
  type Review,
  type ReviewQueryParam
} from '@/apis/review'

const { t } = useI18n()

const listQuery = ref<ReviewQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<Review[]>([])
const listLoading = ref(true)
const total = ref(0)

const viewDialogVisible = ref(false)
const replyDialogVisible = ref(false)
const viewLoading = ref(false)
const replyLoading = ref(false)
const currentReview = ref<Review>({})

const statusOptions = [
  { value: 0, label: 'Pending' },
  { value: 1, label: 'Approved' },
  { value: 2, label: 'Rejected' },
]

const getStatusLabel = (status: number) => {
  switch (status) {
    case 0: return 'Pending'
    case 1: return 'Approved'
    case 2: return 'Rejected'
    default: return 'Unknown'
  }
}

const getStatusType = (status: number) => {
  switch (status) {
    case 0: return 'warning'
    case 1: return 'success'
    case 2: return 'danger'
    default: return 'info'
  }
}

const fetchList = async () => {
  listLoading.value = true
  try {
    const response = await getReviewListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error: any) {
    listLoading.value = false
    ElMessage.error(error.message || 'Failed to fetch reviews')
  }
}

const handleSearch = () => {
  listQuery.value.pageNum = 1
  fetchList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  fetchList()
}

const handleView = async (row: Review) => {
  viewLoading.value = true
  viewDialogVisible.value = true
  try {
    const response = await getReviewByIdAPI(row.id!)
    viewLoading.value = false
    currentReview.value = response.data || {}
  } catch (error: any) {
    viewLoading.value = false
    ElMessage.error(error.message || 'Failed to fetch review details')
  }
}

const handleApprove = async (row: Review) => {
  try {
    await ElMessageBox.confirm('Approve this review?', 'Confirm', {
      confirmButtonText: 'Yes',
      cancelButtonText: 'Cancel',
      type: 'warning',
    })
    await approveReviewAPI(row.id!)
    ElMessage.success('Review approved')
    fetchList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(error.message || 'Failed to approve review')
    }
  }
}

const handleReject = async (row: Review) => {
  try {
    await ElMessageBox.confirm('Reject this review?', 'Confirm', {
      confirmButtonText: 'Yes',
      cancelButtonText: 'Cancel',
      type: 'warning',
    })
    await rejectReviewAPI(row.id!)
    ElMessage.success('Review rejected')
    fetchList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(error.message || 'Failed to reject review')
    }
  }
}

const replyContent = ref('')

const handleReply = (row: Review) => {
  currentReview.value = { ...row }
  replyContent.value = row.replyContent || ''
  replyDialogVisible.value = true
}

const submitReply = async () => {
  if (!replyContent.value.trim()) {
    ElMessage.warning('Please enter reply content')
    return
  }
  replyLoading.value = true
  try {
    await replyReviewAPI({
      reviewId: currentReview.value.id!,
      content: replyContent.value,
    })
    ElMessage.success('Reply submitted')
    replyDialogVisible.value = false
    fetchList()
  } catch (error: any) {
    ElMessage.error(error.message || 'Failed to submit reply')
  } finally {
    replyLoading.value = false
  }
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageSize = val
  fetchList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  fetchList()
}

const formatDate = (dateStr?: string) => {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleString()
}

const renderRating = (rating: number) => {
  return '★'.repeat(rating) + '☆'.repeat(5 - rating)
}

onMounted(() => {
  fetchList()
})
</script>

<template>
  <div class="review-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="listQuery" class="filter-form">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="All" clearable style="width: 150px">
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Product ID">
          <el-input v-model="listQuery.productId" placeholder="Product ID" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item label="Store ID">
          <el-input v-model="listQuery.storeId" placeholder="Store ID" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="productId" label="Product ID" width="100" />
        <el-table-column prop="userNickname" label="User" width="120" />
        <el-table-column label="Rating" width="120">
          <template #default="{ row }">
            <span class="rating">{{ renderRating(row.rating || 0) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="content" label="Content" min-width="200" show-overflow-tooltip />
        <el-table-column prop="likeCount" label="Likes" width="80" />
        <el-table-column prop="replyCount" label="Replies" width="80" />
        <el-table-column label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status || 0)">
              {{ getStatusLabel(row.status || 0) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Create Time" width="160">
          <template #default="{ row }">
            {{ formatDate(row.createTime) }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" :icon="View" @click="handleView(row)">View</el-button>
            <el-button link type="success" :icon="Check" @click="handleApprove(row)" v-if="row.status === 0">Approve</el-button>
            <el-button link type="danger" :icon="Close" @click="handleReject(row)" v-if="row.status === 0">Reject</el-button>
            <el-button link type="warning" :icon="ChatLineRound" @click="handleReply(row)">Reply</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="listQuery.pageNum"
        v-model:page-size="listQuery.pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
        style="margin-top: 20px"
      />
    </el-card>

    <el-dialog v-model="viewDialogVisible" title="Review Details" width="600px">
      <div v-loading="viewLoading">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ID">{{ currentReview.id }}</el-descriptions-item>
          <el-descriptions-item label="Product ID">{{ currentReview.productId }}</el-descriptions-item>
          <el-descriptions-item label="User">{{ currentReview.userNickname }}</el-descriptions-item>
          <el-descriptions-item label="Store">{{ currentReview.storeName }}</el-descriptions-item>
          <el-descriptions-item label="Rating" :span="2">
            <span class="rating">{{ renderRating(currentReview.rating || 0) }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="Content" :span="2">{{ currentReview.content }}</el-descriptions-item>
          <el-descriptions-item label="Images" :span="2">{{ currentReview.images || '-' }}</el-descriptions-item>
          <el-descriptions-item label="Likes">{{ currentReview.likeCount }}</el-descriptions-item>
          <el-descriptions-item label="Replies">{{ currentReview.replyCount }}</el-descriptions-item>
          <el-descriptions-item label="Status">
            <el-tag :type="getStatusType(currentReview.status || 0)">
              {{ getStatusLabel(currentReview.status || 0) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Create Time">{{ formatDate(currentReview.createTime) }}</el-descriptions-item>
          <el-descriptions-item label="Reply" :span="2">{{ currentReview.replyContent || '-' }}</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-dialog>

    <el-dialog v-model="replyDialogVisible" title="Reply to Review" width="500px">
      <el-form label-width="100px">
        <el-form-item label="User">
          <span>{{ currentReview.userNickname }}</span>
        </el-form-item>
        <el-form-item label="Original Content">
          <div class="original-content">{{ currentReview.content }}</div>
        </el-form-item>
        <el-form-item label="Reply Content">
          <el-input
            v-model="replyContent"
            type="textarea"
            :rows="4"
            placeholder="Enter your reply"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="replyDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="replyLoading" @click="submitReply">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.review-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.rating {
  color: #f7ba2a;
  letter-spacing: 2px;
}

.original-content {
  padding: 10px;
  background: #f5f7fa;
  border-radius: 4px;
  max-height: 100px;
  overflow-y: auto;
}

.table-card {
  margin-bottom: 20px;
}
</style>