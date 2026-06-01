<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus } from '@element-plus/icons-vue'
import {
  getAnchorListAPI,
  getAnchorByIdAPI,
  createAnchorAPI,
  type Anchor,
  type AnchorQueryParam
} from '@/apis/live'

const { t } = useI18n()

const listQuery = ref<AnchorQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<Anchor[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogLoading = ref(false)

const formData = ref({
  userId: 0,
  anchorName: '',
  avatar: '',
  description: ''
})

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getAnchorListAPI(listQuery.value)
    listLoading.value = false
    list.value = (response as any).data || []
    total.value = (response as any).total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load anchors')
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => {
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleAdd = () => {
  formData.value = {
    userId: 0,
    anchorName: '',
    avatar: '',
    description: ''
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createAnchorAPI(formData.value)
    ElMessage.success('Anchor created successfully')
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to create anchor')
  } finally {
    dialogLoading.value = false
  }
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return time.replace('T', ' ').substring(0, 19)
}
</script>

<template>
  <div class="anchor-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Anchor Name">
          <el-input v-model="listQuery.anchorName" placeholder="Anchor name" clearable />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" @click="handleAdd">Add Anchor</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="userId" label="User ID" width="100" />
        <el-table-column prop="anchorName" label="Anchor Name" min-width="150" />
        <el-table-column prop="avatar" label="Avatar" width="80">
          <template #default="{ row }">
            <el-avatar v-if="row.avatar" :src="row.avatar" :size="40" />
            <el-avatar v-else :size="40">{{ row.anchorName?.charAt(0) }}</el-avatar>
          </template>
        </el-table-column>
        <el-table-column prop="followerCount" label="Followers" width="100" />
        <el-table-column prop="description" label="Description" min-width="200" show-overflow-tooltip />
        <el-table-column prop="createdAt" label="Created" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="Add Anchor" width="500px" :close-on-click-modal="false">
      <el-form v-loading="dialogLoading" :model="formData" label-width="120px">
        <el-form-item label="User ID">
          <el-input-number v-model="formData.userId" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Anchor Name">
          <el-input v-model="formData.anchorName" />
        </el-form-item>
        <el-form-item label="Avatar">
          <el-input v-model="formData.avatar" placeholder="Image URL" />
        </el-form-item>
        <el-form-item label="Description">
          <el-input v-model="formData.description" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.anchor-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}
</style>