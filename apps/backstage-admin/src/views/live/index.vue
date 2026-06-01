<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, View, VideoPlay, VideoPause } from '@element-plus/icons-vue'
import {
  getLiveRoomListAPI,
  getLiveRoomByIdAPI,
  startLiveRoomAPI,
  endLiveRoomAPI,
  type LiveRoom,
  type LiveQueryParam
} from '@/apis/live'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<LiveQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<LiveRoom[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const isEdit = ref(false)
const viewMode = ref(false)

const formData = ref<LiveRoom>({
  id: undefined,
  anchorId: 0,
  anchorName: '',
  roomName: '',
  coverImage: '',
  description: '',
  status: 'PENDING',
  viewerCount: 0,
  startTime: '',
  endTime: ''
})

const statusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Live', value: 'LIVE' },
  { label: 'Ended', value: 'ENDED' },
  { label: 'Cancelled', value: 'CANCELLED' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getLiveRoomListAPI(listQuery.value)
    listLoading.value = false
    list.value = (response as any).data || []
    total.value = (response as any).total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load live rooms')
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
  isEdit.value = false
  viewMode.value = false
  formData.value = {
    id: undefined,
    anchorId: 0,
    anchorName: '',
    roomName: '',
    coverImage: '',
    description: '',
    status: 'PENDING',
    viewerCount: 0,
    startTime: '',
    endTime: ''
  }
  dialogVisible.value = true
}

const handleView = async (row: LiveRoom) => {
  if (!row.id) return
  try {
    const response = await getLiveRoomByIdAPI(row.id)
    formData.value = response as any || row
    isEdit.value = false
    viewMode.value = true
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load room details')
  }
}

const handleStartLive = async (row: LiveRoom) => {
  if (!row.id || !row.anchorId) return
  try {
    await ElMessageBox.confirm(`Start live streaming for ${row.roomName}?`, 'Confirm', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await startLiveRoomAPI({
      anchorId: row.anchorId,
      roomName: row.roomName,
      coverImage: row.coverImage,
      description: row.description
    })
    ElMessage.success('Live room started')
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to start live room')
    }
  }
}

const handleEndLive = async (row: LiveRoom) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm(`End live streaming for ${row.roomName}?`, 'Confirm', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await endLiveRoomAPI(row.id)
    ElMessage.success('Live room ended')
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to end live room')
    }
  }
}

const getStatusType = (status: string): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    'PENDING': 'info',
    'LIVE': 'success',
    'ENDED': 'warning',
    'CANCELLED': 'danger'
  }
  return map[status]
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return formatDateTime(time)
}
</script>

<template>
  <div class="live-room-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Room Name">
          <el-input v-model="listQuery.roomName" placeholder="Room name" clearable />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" @click="handleAdd">Add Room</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="roomName" label="Room Name" min-width="150" />
        <el-table-column prop="anchorName" label="Anchor" width="120" />
        <el-table-column prop="viewerCount" label="Viewers" width="100" />
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="startTime" label="Start Time" width="180">
          <template #default="{ row }">
            {{ formatTime(row.startTime) }}
          </template>
        </el-table-column>
        <el-table-column prop="endTime" label="End Time" width="180">
          <template #default="{ row }">
            {{ formatTime(row.endTime) }}
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">View</el-button>
            <el-button v-if="row.status === 'PENDING'" link type="success" size="small" :icon="VideoPlay" @click="handleStartLive(row)">Start</el-button>
            <el-button v-if="row.status === 'LIVE'" link type="warning" size="small" :icon="VideoPause" @click="handleEndLive(row)">End</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="viewMode ? 'Room Details' : (isEdit ? 'Edit Room' : 'Add Room')"
      width="600px"
      :close-on-click-modal="false"
    >
      <el-form v-loading="dialogLoading" :model="formData" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Room Name">
              <el-input v-model="formData.roomName" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Anchor ID">
              <el-input-number v-model="formData.anchorId" :min="1" :disabled="viewMode" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Cover Image">
              <el-input v-model="formData.coverImage" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Status">
              <el-select v-model="formData.status" :disabled="viewMode" style="width: 100%">
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :disabled="viewMode" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.live-room-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}
</style>