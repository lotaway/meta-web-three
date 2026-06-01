<script setup lang=\"ts\">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, Check, Close, View } from '@element-plus/icons-vue'
import {
  getReturnApplyListAPI,
  returnApplyUpdateStatusAPI,
  getReturnApplyByIdAPI,
  type OmsOrderReturnApply,
  type ReturnApplyQueryParam
} from '@/apis/returnApply'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<ReturnApplyQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<OmsOrderReturnApply[]>([])
const listLoading = ref(true)
const total = ref(0)

const detailDialogVisible = ref(false)
const detailData = ref<OmsOrderReturnApply | null>(null)

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getReturnApplyListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data.list || []
    total.value = response.data.total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error(t('payment.refund.list.loadFailed'))
  }
}

onMounted(() => {
  getList()
})

const handleSelectionChange = (val: OmsOrderReturnApply[]) => {
  multipleSelection.value = val
}

const multipleSelection = ref<OmsOrderReturnApply[]>([])

const handleSearch = () => {
  listQuery.value.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

const handleView = async (row: OmsOrderReturnApply) => {
  try {
    const response = await getReturnApplyByIdAPI(row.id)
    detailData.value = response.data
    detailDialogVisible.value = true
  } catch (error) {
    ElMessage.error(t('payment.refund.detail.loadFailed'))
  }
}

const handleApprove = async (row: OmsOrderReturnApply) => {
  try {
    await ElMessageBox.confirm(
      t('payment.refund.approve.confirm'),
      t('common.verify'),
      {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
      }
    )
    await returnApplyUpdateStatusAPI(row.id, { status: 2 })
    ElMessage.success(t('payment.refund.approve.success'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('payment.refund.approve.failed'))
    }
  }
}

const handleReject = async (row: OmsOrderReturnApply) => {
  try {
    await ElMessageBox.confirm(
      t('payment.refund.reject.confirm'),
      t('common.verify'),
      {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
      }
    )
    await returnApplyUpdateStatusAPI(row.id, { status: 3 })
    ElMessage.success(t('payment.refund.reject.success'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('payment.refund.reject.failed'))
    }
  }
}

const statusOptions = [
  { label: t('payment.refund.status.pending'), value: 0 },
  { label: t('payment.refund.status.processing'), value: 1 },
  { label: t('payment.refund.status.approved'), value: 2 },
  { label: t('payment.refund.status.rejected'), value: 3 },
  { label: t('payment.refund.status.completed'), value: 4 }
]

const getStatusLabel = (status: number) => {
  const option = statusOptions.find(opt => opt.value === status)
  return option ? option.label : '-'
}
</script>

<template>
  <div class=\"app-container\">
    <el-card class=\"search-card\" shadow=\"never\">
      <div class=\"search-form\">
        <el-form :inline=\"true\" :model=\"listQuery\" class=\"demo-form-inline\">
          <el-form-item :label=\"t('payment.refund.list.orderNo')\">
            <el-input
              v-model=\"listQuery.orderNo\"
              :placeholder=\"t('payment.refund.list.orderNoPlaceholder')\"
              clearable
              @keyup.enter=\"handleSearch\"
            />
          </el-form-item>
          <el-form-item :label=\"t('payment.refund.list.status')\">
            <el-select v-model=\"listQuery.status\" :placeholder=\"t('common.select')\" clearable>
              <el-option
                v-for=\"item in statusOptions\"
                :key=\"item.value\"
                :label=\"item.label\"
                :value=\"item.value\"
              />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type=\"primary\" :icon=\"Search\" @click=\"handleSearch\">
              {{ t('common.search') }}
            </el-button>
            <el-button :icon=\"Refresh\" @click=\"handleReset\">
              {{ t('common.reset') }}
            </el-button>
          </el-form-item>
        </el-form>
      </div>
    </el-card>

    <el-card class=\"table-card\" shadow=\"never\">
      <template #header>
        <div class=\"card-header\">
          <span>{{ t('payment.refund.list.title') }}</span>
        </div>
      </template>

      <el-table
        v-loading=\"listLoading\"
        :data=\"list\"
        border
        stripe
        @selection-change=\"handleSelectionChange\"
      >
        <el-table-column type=\"selection\" width=\"55\" align=\"center\" />
        <el-table-column label=\"ID\" prop=\"id\" width=\"80\" align=\"center\" />
        <el-table-column :label=\"t('payment.refund.list.orderNo')\" prop=\"orderNo\" min-width=\"150\" />
        <el-table-column :label=\"t('payment.refund.list.productName')\" prop=\"productName\" min-width=\"180\" />
        <el-table-column :label=\"t('payment.refund.list.refundAmount')\" prop=\"refundAmount\" width=\"120\" align=\"right\">
          <template #default=\"{ row }\">
            ¥{{ row.refundAmount?.toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column :label=\"t('payment.refund.list.refundReason')\" prop=\"refundReason\" min-width=\"150\" show-overflow-tooltip />
        <el-table-column :label=\"t('payment.refund.list.status')\" prop=\"status\" width=\"120\" align=\"center\">
          <template #default=\"{ row }\">
            <el-tag :type=\"row.status === 2 ? 'success' : row.status === 3 ? 'danger' : 'warning'\">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label=\"t('payment.refund.list.createTime')\" prop=\"createTime\" width=\"160\">
          <template #default=\"{ row }\">
            {{ formatDateTime(row.createTime) }}
          </template>
        </el-table-column>
        <el-table-column :label=\"t('common.action')\" width=\"200\" align=\"center\" fixed=\"right\">
          <template #default=\"{ row }\">
            <el-button link type=\"primary\" :icon=\"View\" @click=\"handleView(row)\">
              {{ t('common.view') }}
            </el-button>
            <el-button
              v-if=\"row.status === 0 || row.status === 1\"
              link
              type=\"success\"
              :icon=\"Check\"
              @click=\"handleApprove(row)\"
            >
              {{ t('payment.refund.approve.button') }}
            </el-button>
            <el-button
              v-if=\"row.status === 0 || row.status === 1\"
              link
              type=\"danger\"
              :icon=\"Close\"
              @click=\"handleReject(row)\"
            >
              {{ t('payment.refund.reject.button') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class=\"pagination-container\">
        <el-pagination
          v-model:current-page=\"listQuery.pageNum\"
          v-model:page-size=\"listQuery.pageSize\"
          :page-sizes=\"[10, 20, 50, 100]\"
          :total=\"total\"
          layout=\"total, sizes, prev, pager, next, jumper\"
          @size-change=\"handleSizeChange\"
          @current-change=\"handleCurrentChange\"
        />
      </div>
    </el-card>

    <el-dialog v-model=\"detailDialogVisible\" title=\"Refund Detail\" width=\"600px\">
      <el-descriptions v-if=\"detailData\" :column=\"1\" border>
        <el-descriptions-item :label=\"t('payment.refund.detail.id')\">
          {{ detailData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.orderNo')\">
          {{ detailData.orderNo }}
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.productName')\">
          {{ detailData.productName }}
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.refundAmount')\">
          ¥{{ detailData.refundAmount?.toFixed(2) }}
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.refundReason')\">
          {{ detailData.refundReason }}
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.status')\">
          <el-tag :type=\"detailData.status === 2 ? 'success' : detailData.status === 3 ? 'danger' : 'warning'\">
            {{ getStatusLabel(detailData.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label=\"t('payment.refund.detail.createTime')\">
          {{ formatDateTime(detailData.createTime) }}
        </el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click=\"detailDialogVisible = false\">{{ t('common.close') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.search-card {
  margin-bottom: 20px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>