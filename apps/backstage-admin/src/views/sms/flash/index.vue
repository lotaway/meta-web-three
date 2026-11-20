<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getFlashListAPI, flashUpdateStatusByIdAPI, flashDeleteByIdAPI, flashCreateAPI, flashUpdateByIdAPI } from '@/apis/flash'
import { formatDate } from '@/utils/datetime'
import type { SmsFlashPromotion } from '@/types/flash'
import type { PageParam } from '@/types/common'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route
const router = useRouter()

// List query params
const listQuery = ref<PageParam>({
  pageNum: 1,
  pageSize: 10,
  keyword: ''
})
// List data
const list = ref<SmsFlashPromotion[]>([])
// Total count
const total = ref<number>(0)
// Load images
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getFlashListAPI(listQuery.value)
    listLoading.value = false
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load flash sale list:', error)
  }
}

// Fetch data on mount
onMounted(() => {
  getList()
})

// Default flash sale activity object
const defaultSmsFlashPromotion: SmsFlashPromotion = {
  title: '',
  status: 0
}
// Current flash sale activity object
const flashPromotion = ref(Object.assign({}, defaultSmsFlashPromotion))
// Whether the edit dialog is visible
const dialogVisible = ref(false)
// Whether in edit state
const isEdit = ref(false)

// Reset search conditions
const handleResetSearch = () => {
  listQuery.value = { pageNum: 0, pageSize: 10 }
}

// Search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Add activity
const handleAdd = () => {
  dialogVisible.value = true
  isEdit.value = false
  flashPromotion.value = Object.assign({}, defaultSmsFlashPromotion)
}

// Show time slot list
const handleShowSessionList = () => {
  router.push({ path: '/sms/flashSession' })
}

// Status change
const handleStatusChange = async (index: number, row: SmsFlashPromotion) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeStatus'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await flashUpdateStatusByIdAPI(row.id!, { status: row.status })
    ElMessage.success(t('message.modifySuccess'))
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to update status:', error)
    }
    ElMessage.info(t('message.cancelled'))
    getList()
  }
}

// Delete activity
const handleDelete = async (index: number, row: SmsFlashPromotion) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteActivity'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await flashDeleteByIdAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete activity:', error)
    }
  }
}

// Update activity
const handleUpdate = (index: number, row: SmsFlashPromotion) => {
  dialogVisible.value = true
  isEdit.value = true
  flashPromotion.value = Object.assign({}, row)
}

const saveFlashPromotion = async () => {
  if (isEdit.value) {
    await flashUpdateByIdAPI(flashPromotion.value.id!, flashPromotion.value)
    return t('message.modifySuccess')
  }
  await flashCreateAPI(flashPromotion.value)
  return t('message.operationSuccess')
}

// Handle dialog confirmation
const handleDialogConfirm = async () => {
  await ElMessageBox.confirm(t('dialog.confirmAction'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  try {
    ElMessage.success(await saveFlashPromotion())
    dialogVisible.value = false
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to process activity:', error)
    }
  }
}

// Select time slot
const handleSelectSession = (index: number, row: SmsFlashPromotion) => {
  router.push({ path: '/sms/selectSession', query: { flashPromotionId: row.id } })
}

// Format activity status
const formatActiveStatus = (row: SmsFlashPromotion) => {
  const nowDate = new Date()
  const startDate = new Date(row.startDate!)
  const endDate = new Date(row.endDate!)
  if (nowDate.getTime() >= startDate.getTime() && nowDate.getTime() <= endDate.getTime()) {
    return '活动进行中'
  } else if (nowDate.getTime() > endDate.getTime()) {
    return '活动已结束'
  } else {
    return '活动未开始'
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
      <div style="margin-top: 15px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item :label="t('groupBuying.activityName')">
            <el-input v-model="listQuery.keyword" class="input-width" :placeholder="t('groupBuying.activityName')" clearable></el-input>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="handleAdd()" style="margin-left: 20px">{{ t('promotion.addActivity') }}</el-button>
      <el-button class="btn-add" @click="handleShowSessionList()">{{ t('promotion.seckillSlots') }}</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="flashTable" :data="list" style="width: 100%;" v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="活动标题" align="center">
          <template #default="scope">{{ scope.row.title }}</template>
        </el-table-column>
        <el-table-column label="活动状态" width="140" align="center">
          <template #default="scope">{{ formatActiveStatus(scope.row as SmsFlashPromotion) }}</template>
        </el-table-column>
        <el-table-column label="开始时间" width="140" align="center">
          <template #default="scope">{{ formatDate(scope.row.startDate) }}</template>
        </el-table-column>
        <el-table-column label="结束时间" width="140" align="center">
          <template #default="scope">{{ formatDate(scope.row.endDate) }}</template>
        </el-table-column>
        <el-table-column label="上线/下线" width="200" align="center">
          <template #default="scope">
            <el-switch @change="handleStatusChange(scope.$index, scope.row as SmsFlashPromotion)" :active-value="1" :inactive-value="0"
              v-model="scope.row.status">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleSelectSession(scope.$index, scope.row as SmsFlashPromotion)">{{ t('promotion.setProducts') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleUpdate(scope.$index, scope.row as SmsFlashPromotion)">
              {{ t('common.edit') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as SmsFlashPromotion)">{{ t('common.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" v-model:current-page="listQuery.pageNum"
        :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="total">
      </el-pagination>
    </div>
    <el-dialog title="添加活动" v-model="dialogVisible" width="40%">
      <el-form :model="flashPromotion" ref="SmsFlashPromotionForm" label-width="150px">
        <el-form-item label="活动标题：">
          <el-input v-model="flashPromotion.title" style="width: 250px"></el-input>
        </el-form-item>
        <el-form-item label="开始时间：">
          <el-date-picker v-model="flashPromotion.startDate" type="date" placeholder="请选择时间">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="结束时间：">
          <el-date-picker v-model="flashPromotion.endDate" type="date" placeholder="请选择时间">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="上线/下线">
          <el-radio-group v-model="flashPromotion.status">
            <el-radio :label="1">上线</el-radio>
            <el-radio :label="0">下线</el-radio>
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

<style></style>
