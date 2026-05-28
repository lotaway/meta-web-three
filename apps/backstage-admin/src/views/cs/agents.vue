<template>
  <div class="cs-agents">
    <el-card>
      <template #header>
        <div class="card-header"><span>{{ t('cs.agents.title') }}</span><el-button type="primary" size="small" @click="showDialog = true">{{ t('cs.agents.addAgent') }}</el-button></div>
      </template>
      <el-table :data="agents" border stripe>
        <el-table-column prop="id" :label="t('cs.agents.id')" width="60" />
        <el-table-column prop="adminId" :label="t('cs.agents.adminId')" width="100" />
        <el-table-column prop="nickname" :label="t('cs.agents.nickname')" />
        <el-table-column prop="status" :label="t('cs.agents.status')" width="90">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="currentLoad" :label="t('cs.agents.currentLoad')" width="90" />
        <el-table-column prop="maxConcurrent" :label="t('cs.agents.maxConcurrent')" width="90" />
        <el-table-column prop="groupId" :label="t('cs.agents.groupId')" width="80" />
        <el-table-column :label="t('cs.agents.operations')" width="220">
          <template #default="{ row }">
            <el-button size="small" @click="toggleStatus(row)">{{ row.status === 'ONLINE' ? t('cs.agents.offline') : t('cs.agents.online') }}</el-button>
            <el-button size="small" type="danger" @click="handleDelete(row.id)">{{ t('cs.agents.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    <el-dialog v-model="showDialog" :title="t('cs.agents.dialogTitle')" width="400px">
      <el-form :model="form" label-width="80px">
        <el-form-item :label="t('cs.agents.adminIdLabel')"><el-input v-model.number="form.adminId" type="number" /></el-form-item>
        <el-form-item :label="t('cs.agents.nicknameLabel')"><el-input v-model="form.nickname" /></el-form-item>
        <el-form-item :label="t('cs.agents.groupIdLabel')"><el-input v-model.number="form.groupId" type="number" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">{{ t('cs.agents.cancel') }}</el-button>
        <el-button type="primary" @click="handleCreate">{{ t('cs.agents.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { getOnlineAgentsAPI, agentOnlineAPI, agentOfflineAPI, deleteAgentAPI } from '@/apis/cs'
import type { Agent } from '@/apis/cs'
import { ElMessage } from 'element-plus'
import http from '@/utils/http'

const { t } = useI18n()

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
  ElMessage.success(t('cs.agents.operationSuccess'))
}

const handleDelete = async (id: number) => {
  await deleteAgentAPI(id)
  ElMessage.success(t('cs.agents.deleted'))
  load()
}

const handleCreate = async () => {
  await http({ url: '/cs/agent/create', method: 'post', data: form.value })
  ElMessage.success(t('cs.agents.createSuccess'))
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