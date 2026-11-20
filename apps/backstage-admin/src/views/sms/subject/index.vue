<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getHomeRecommendSubjectListAPI, homeRecommendSubjectUpdateRecommendStatusAPI, homeRecommendSubjectDeleteByIdsAPI, homeRecommendSubjectCreateAPI, homeRecommendSubjectUpdateSortAPI } from '@/apis/homeSubject'
import { getSubjectListAPI } from '@/apis/subject'
import { formatDateTime } from '@/utils/datetime'
import type { HomeSubjectQueryParam, SmsHomeRecommendSubject } from '@/types/homeSubject'
import type { CmsSubject } from '@/types/subject'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()


// List query params
const listQuery = ref<HomeSubjectQueryParam>({
  pageNum: 1,
  pageSize: 10
})
// Query recommendation status
const recommendOptions = ref([
  {
    label: '未推荐',
    value: 0
  },
  {
    label: '推荐中',
    value: 1
  }
])
// List data
const list = ref<SmsHomeRecommendSubject[]>([])
// Total count
const total = ref(0)
// Loading state
const listLoading = ref(false)
// Get list data
const getList = async () => {
  listLoading.value = true
  const res = await getHomeRecommendSubjectListAPI(listQuery.value)
  listLoading.value = false
  list.value = res.data.list
  total.value = res.data.total
}

// Run after component mounts
onMounted(() => {
  getList()
})

// Batch operation selected records
const multipleSelection = ref<SmsHomeRecommendSubject[]>([])
// Batch operations
const operates = ref([
  {
    label: "设为推荐",
    value: 0
  },
  {
    label: "取消推荐",
    value: 1
  },
  {
    label: "删除",
    value: 2
  }
])
// Current operation type
const operateType = ref()

// Select-subject dialog visibility
const selectDialogVisible = ref(false)
// Select-subject dialog data
const dialogData = ref({
  list: [] as CmsSubject[],
  total: 0,
  multipleSelection: [] as CmsSubject[],
  listQuery: {
    keyword: '',
    pageNum: 1,
    pageSize: 5
  }
})

// Set sort dialog visibility
const sortDialogVisible = ref(false)
// Set sort dialog data
const sortDialogData = ref({ sort: 0, id: 0 })

// Reset search
const handleResetSearch = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
}

// Search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Handle table selection change
const handleSelectionChange = (val: SmsHomeRecommendSubject[]) => {
  multipleSelection.value = val
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

// Handle recommendation status change
const handleRecommendStatusStatusChange = (index: number, row: SmsHomeRecommendSubject) => {
  updateRecommendStatusStatus([row.id!], row.recommendStatus!)
}

// Handle delete
const handleDelete = async (index: number, row: SmsHomeRecommendSubject) => {
  deleteSubject([row.id!])
}

// Handle batch operations
const batchOperateHandlers: Record<number, (ids: number[]) => Promise<void>> = {
  0: ids => updateRecommendStatusStatus(ids, 1),
  1: ids => updateRecommendStatusStatus(ids, 0),
  2: ids => deleteSubject(ids)
}

const showBatchWarning = (message: string) => ElMessage.warning({ message, duration: 1000 })

const handleBatchOperate = async () => {
  if (multipleSelection.value.length < 1) {
    showBatchWarning('请选择一条记录')
    return
  }
  const handler = batchOperateHandlers[operateType.value]
  if (!handler) {
    showBatchWarning('请选择批量操作类型')
    return
  }
  await handler(multipleSelection.value.map(item => item.id!))
}

// Handle subject selection
const handleSelectSubject = () => {
  selectDialogVisible.value = true
  dialogData.value.listQuery.keyword = ''
  getDialogList()
}

// Handle selection search
const handleSelectSearch = () => {
  getDialogList()
}

// Handle dialog page size change
const handleDialogSizeChange = (val: number) => {
  dialogData.value.listQuery.pageNum = 1
  dialogData.value.listQuery.pageSize = val
  getDialogList()
}

// Handle dialog current page change
const handleDialogCurrentChange = (val: number) => {
  dialogData.value.listQuery.pageNum = val
  getDialogList()
}

// Handle dialog selection change
const handleDialogSelectionChange = (val: CmsSubject[]) => {
  dialogData.value.multipleSelection = val
}

// Handle selection dialog confirmation
const buildSubjectPayload = () => dialogData.value.multipleSelection.map(item => ({
  subjectId: item.id!,
  subjectName: item.title!
}))

const handleSelectDialogConfirm = async () => {
  if (dialogData.value.multipleSelection.length < 1) {
    ElMessage.warning({ message: '请选择一条记录', duration: 1000 })
    return
  }
  await ElMessageBox.confirm(t('dialog.confirmAdd'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await homeRecommendSubjectCreateAPI(buildSubjectPayload())
  selectDialogVisible.value = false
  dialogData.value.multipleSelection = []
  getList()
  ElMessage.success({ type: 'success', message: '添加成功!' })
}

// Handle edit sort
const handleEditSort = (index: number, row: SmsHomeRecommendSubject) => {
  sortDialogVisible.value = true
  sortDialogData.value.sort = row.sort!
  sortDialogData.value.id = row.id!
}

// Handle update sort
const handleUpdateSort = async () => {
  await ElMessageBox.confirm(t('dialog.confirmChangeSort'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await homeRecommendSubjectUpdateSortAPI(sortDialogData.value)
  sortDialogVisible.value = false
  getList()
  ElMessage.success({
    type: 'success',
    message: '修改成功!'
  })
}

// Update recommendation status
const updateRecommendStatusStatus = async (ids: number[], status: number) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmChangeRecommend'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await homeRecommendSubjectUpdateRecommendStatusAPI({ ids: ids.join(','), recommendStatus: status })
    getList()
    ElMessage.success({
      type: 'success',
      message: '修改成功!'
    })
  } catch {
    getList()
  }
}

// Delete subject
const deleteSubject = async (ids: number[]) => {
  await ElMessageBox.confirm(t('dialog.confirmDeleteRecommend'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await homeRecommendSubjectDeleteByIdsAPI({ ids: ids.join(',') })
  getList()
  ElMessage.success({
    type: 'success',
    message: '删除成功!'
  })
}

// Get dialog list
const getDialogList = async () => {
  const res = await getSubjectListAPI(dialogData.value.listQuery)
  dialogData.value.list = res.data.list
  dialogData.value.total = res.data.total
}

// Format recommendation status
const formatRecommendStatus = (status: number) => {
  if (status === 1) {
    return '推荐中'
  } else {
    return '未推荐'
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
          <el-form-item :label="t('promotion.subjectName')">
            <el-input v-model="listQuery.subjectName" class="input-width" :placeholder="t('promotion.subjectNamePlain')"></el-input>
          </el-form-item>
          <el-form-item :label="t('promotion.recommendStatus')">
            <el-select v-model="listQuery.recommendStatus" :placeholder="t('product.all')" clearable class="input-width"
              style="width: 140px;">
              <el-option v-for="item in recommendOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="handleSelectSubject()">{{ t('promotion.selectSubject') }}</el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="newSubjectTable" :data="list" style="width: 100%" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="120" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="专题名称" align="center">
          <template #default="scope">{{ scope.row.subjectName }}</template>
        </el-table-column>
        <el-table-column label="是否推荐" width="200" align="center">
          <template #default="scope">
            <el-switch @change="handleRecommendStatusStatusChange(scope.$index, scope.row as SmsHomeRecommendSubject)" :active-value="1"
              :inactive-value="0" v-model="scope.row.recommendStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="排序" width="160" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="状态" width="160" align="center">
          <template #default="scope">{{ formatRecommendStatus(scope.row.recommendStatus) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="180" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleEditSort(scope.$index, scope.row as SmsHomeRecommendSubject)">{{ t('promotion.setSort') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row as SmsHomeRecommendSubject)">{{ t('common.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" placeholder="批量操作">
        <el-option v-for="item in operates" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        确定
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
    <el-dialog v-model="selectDialogVisible" title="选择专题" width="50%">
      <el-input v-model="dialogData.listQuery.keyword" style="width: 250px;margin-bottom: 20px" placeholder="专题名称搜索">
        <template #append>
          <el-button :icon="Search" @click="handleSelectSearch()"></el-button>
        </template>
      </el-input>
      <el-table :data="dialogData.list" @selection-change="handleDialogSelectionChange" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column label="专题名称" align="center">
          <template #default="scope">{{ scope.row.title }}</template>
        </el-table-column>
        <el-table-column label="所属分类" width="160" align="center">
          <template #default="scope">{{ scope.row.categoryName }}</template>
        </el-table-column>
        <el-table-column label="添加时间" width="160" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.createTime) }}</template>
        </el-table-column>
      </el-table>
      <div class="pagination-container">
        <el-pagination background @size-change="handleDialogSizeChange" @current-change="handleDialogCurrentChange"
          layout="prev, pager, next" v-model:current-page="dialogData.listQuery.pageNum"
          :page-size="dialogData.listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="dialogData.total">
        </el-pagination>
      </div>
      <div style="clear: both"></div>
      <template #footer>
        <div>
          <el-button @click="selectDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleSelectDialogConfirm()">{{ t('common.confirm') }}</el-button>
        </div>
      </template>
    </el-dialog>
    <el-dialog v-model="sortDialogVisible" title="设置排序" width="40%">
      <el-form :model="sortDialogData" label-width="150px">
        <el-form-item label="排序：">
          <el-input v-model="sortDialogData.sort" style="width: 200px"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span>
          <el-button @click="sortDialogVisible = false" size="small">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleUpdateSort" size="small">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style></style>
