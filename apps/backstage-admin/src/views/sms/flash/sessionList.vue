<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Tickets } from '@element-plus/icons-vue'
import { getFlashSessionListAPI, flashSessionUpdateStatusByIdAPI, flashSessionDeleteByIdAPI, flashSessionCreateAPI, flashSessionUpdateByIdAPI } from '@/apis/flashSession'
import type { SmsFlashPromotionSession } from '@/types/flash'
import { formatTime } from '@/utils/datetime'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Flash sale time slot list data
const list = ref<SmsFlashPromotionSession[]>([])
// Loading state
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getFlashSessionListAPI()
    listLoading.value = false
    list.value = res.data
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load flash session list:', error)
  }
}

// Fetch data on mount
onMounted(() => {
  getList()
})

// Default flash sale session object
const defaultFlashSession: SmsFlashPromotionSession = {
  name: '',
  status: 0
}
// Current flash sale session
const flashSession = ref(Object.assign({}, defaultFlashSession))
// Edit dialog display state
const dialogVisible = ref(false)
// Whether in edit mode
const isEdit = ref(false)

// Add time slot
const handleAdd = () => {
  dialogVisible.value = true
  isEdit.value = false
  flashSession.value = Object.assign({}, defaultFlashSession)
}

// Status change
const handleStatusChange = async (index: number, row: SmsFlashPromotionSession) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeStatus'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await flashSessionUpdateStatusByIdAPI(row.id!, { status: row.status })
    ElMessage.success(t('message.modifySuccess'))
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to update status:', error)
    }
    ElMessage.info(t('message.cancelled'))
    getList()
  }
}

// Update time slot
const handleUpdate = (index: number, row: SmsFlashPromotionSession) => {
  dialogVisible.value = true
  isEdit.value = true
  flashSession.value = Object.assign({}, row)
  if (row.startTime) {
    flashSession.value.startTime = row.startTime
  }
  if (row.endTime) {
    flashSession.value.endTime = row.endTime
  }
}

// Delete time slot
const handleDelete = async (index: number, row: SmsFlashPromotionSession) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteTimeSlot'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await flashSessionDeleteByIdAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete time slot:', error)
    }
  }
}

const saveFlashSession = async () => {
  if (isEdit.value) {
    await flashSessionUpdateByIdAPI(flashSession.value.id!, flashSession.value)
    return t('message.modifySuccess')
  }
  await flashSessionCreateAPI(flashSession.value)
  return t('message.operationSuccess')
}

// Handle dialog confirmation
const handleDialogConfirm = async () => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmAction'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    ElMessage.success(await saveFlashSession())
    dialogVisible.value = false
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to process time slot:', error)
    }
  }
}
</script>

<template>
  <div class="app-container">
    <el-card shadow="never" class="operate-container">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="handleAdd()">{{ t('common.add') }}</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="flashSessionTable" :data="list" style="width: 100%;" v-loading="listLoading" border>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="秒杀时间段名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="每日开始时间" align="center">
          <template #default="scope">{{ formatTime(scope.row.startTime) }}</template>
        </el-table-column>
        <el-table-column label="每日结束时间" align="center">
          <template #default="scope">{{ formatTime(scope.row.endTime) }}</template>
        </el-table-column>
        <el-table-column label="启用" align="center">
          <template #default="scope">
            <el-switch @change="handleStatusChange(scope.$index, scope.row as SmsFlashPromotionSession)" :active-value="1" :inactive-value="0"
              v-model="scope.row.status">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleUpdate(scope.$index, scope.row as SmsFlashPromotionSession)">{{ t('common.edit') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as SmsFlashPromotionSession)">{{ t('common.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <el-dialog title="添加时间段" v-model="dialogVisible" width="40%">
      <el-form :model="flashSession" ref="flashSessionForm" label-width="150px">
        <el-form-item label="秒杀时间段名称：">
          <el-input v-model="flashSession.name" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="每日开始时间：">
          <el-time-picker v-model="flashSession.startTime" placeholder="请选择时间">
          </el-time-picker>
        </el-form-item>
        <el-form-item label="每日结束时间：">
          <el-time-picker v-model="flashSession.endTime" placeholder="请选择时间">
          </el-time-picker>
        </el-form-item>
        <el-form-item label="是否启用">
          <el-radio-group v-model="flashSession.status">
            <el-radio :label="1">启用</el-radio>
            <el-radio :label="0">不启用</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleDialogConfirm()">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.operate-container {
  margin-top: 0;
}
</style>
