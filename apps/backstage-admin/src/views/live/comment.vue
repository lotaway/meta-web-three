<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import {
  getLiveRoomListAPI,
  getLiveCommentsByRoomAPI,
  postCommentAPI,
  type LiveRoom,
  type LiveComment
} from '@/apis/live'

const { t } = useI18n()

const roomId = ref<number | undefined>()
const rooms = ref<LiveRoom[]>([])
const comments = ref<LiveComment[]>([])
const loading = ref(false)

const dialogVisible = ref(false)
const dialogLoading = ref(false)

const formData = ref({
  roomId: 0,
  userId: 0,
  userName: '',
  content: ''
})

const getRooms = async () => {
  try {
    const response = await getLiveRoomListAPI({ pageNum: 1, pageSize: 100 })
    rooms.value = (response as any).data || []
  } catch (error) {
    ElMessage.error('Failed to load rooms')
  }
}

const handleSearch = async () => {
  if (!roomId.value) {
    ElMessage.warning('Please select a room')
    return
  }
  loading.value = true
  try {
    const response = await getLiveCommentsByRoomAPI(roomId.value)
    comments.value = response.data || []
    loading.value = false
  } catch (error) {
    loading.value = false
    ElMessage.error('Failed to load comments')
  }
}

const handleAdd = () => {
  formData.value = {
    roomId: roomId.value || 0,
    userId: 0,
    userName: '',
    content: ''
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await postCommentAPI(formData.value)
    ElMessage.success('Comment posted successfully')
    dialogVisible.value = false
    handleSearch()
  } catch (error) {
    ElMessage.error('Failed to post comment')
  } finally {
    dialogLoading.value = false
  }
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return time.replace('T', ' ').substring(0, 19)
}

onMounted(() => {
  getRooms()
})
</script>

<template>
  <div class="comment-container">
    <el-card class="search-card">
      <el-form :inline="true" class="search-form">
        <el-form-item label="Live Room">
          <el-select v-model="roomId" placeholder="Select room" @change="handleSearch">
            <el-option v-for="room in rooms" :key="room.id || 0" :label="room.roomName" :value="room.id || 0" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button type="success" :disabled="roomId === undefined" @click="handleAdd">Post Comment</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="loading" :data="comments" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="userId" label="User ID" width="100" />
        <el-table-column prop="userName" label="Username" width="120" />
        <el-table-column prop="content" label="Content" min-width="300" show-overflow-tooltip />
        <el-table-column prop="createdAt" label="Time" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="Post Comment" width="500px" :close-on-click-modal="false">
      <el-form v-loading="dialogLoading" :model="formData" label-width="100px">
        <el-form-item label="Room ID">
          <el-input v-model="formData.roomId" disabled />
        </el-form-item>
        <el-form-item label="User ID">
          <el-input-number v-model="formData.userId" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Username">
          <el-input v-model="formData.userName" />
        </el-form-item>
        <el-form-item label="Content">
          <el-input v-model="formData.content" type="textarea" :rows="3" />
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
.comment-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}
</style>