<template>
  <div class="cs-agents">
    <el-card>
      <template #header>
        <div class="card-header"><span>客服人员</span><el-button type="primary" size="small" @click="showDialog = true">添加客服</el-button></div>
      </template>
      <el-table :data="agents" border stripe>
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="adminId" label="管理员ID" width="100" />
        <el-table-column prop="nickname" label="昵称" />
        <el-table-column prop="status" label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="currentLoad" label="当前负载" width="90" />
        <el-table-column prop="maxConcurrent" label="最大接待" width="90" />
        <el-table-column prop="groupId" label="技能组" width="80" />
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small" @click="toggleStatus(row)">{{ row.status === 'ONLINE' ? '下线' : '上线' }}</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    <el-dialog v-model="showDialog" title="添加客服" width="400px">
      <el-form :model="form" label-width="80px">
        <el-form-item label="管理员ID"><el-input v-model.number="form.adminId" type="number" /></el-form-item>
        <el-form-item label="昵称"><el-input v-model="form.nickname" /></el-form-item>
        <el-form-item label="技能组"><el-input v-model.number="form.groupId" type="number" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getOnlineAgentsAPI, agentOnlineAPI, agentOfflineAPI } from '@/apis/cs'
import type { Agent } from '@/apis/cs'
import { ElMessage } from 'element-plus'
import http from '@/utils/http'

const agents = ref<Agent[]>([])
const showDialog = ref(false)
const form = ref({ adminId: 0, nickname: '', groupId: 0 })

const load = async () => {
  const res = await getOnlineAgentsAPI()
  agents.value = res.data || []
}

const toggleStatus = async (row: Agent) => {
  if (row.status === 'ONLINE') {
    await agentOfflineAPI(row.id)
    row.status = 'OFFLINE'
  } else {
    await agentOnlineAPI(row.id)
    row.status = 'ONLINE'
  }
  ElMessage.success('操作成功')
}

const handleCreate = async () => {
  await http({ url: '/cs/agent/create', method: 'post', data: form.value })
  ElMessage.success('创建成功')
  showDialog.value = false
  load()
}

const statusTagType = (s: string) => {
  if (s === 'ONLINE') return 'success'
  if (s === 'BUSY') return 'warning'
  if (s === 'AWAY') return 'info'
  return 'danger'
}

onMounted(load)
</script>
