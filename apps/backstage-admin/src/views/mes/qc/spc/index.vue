<template>
  <div class="spc-control-chart-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.spc.chartCode')">
          <el-input v-model="queryParams.chartCode" :placeholder="t('mes.qc.spc.chartCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.spc.chartName')">
          <el-input v-model="queryParams.chartName" :placeholder="t('mes.qc.spc.chartNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.spc.chartType')">
          <el-select v-model="queryParams.chartType" :placeholder="t('mes.qc.spc.chartTypePlaceholder')" clearable>
            <el-option :label="t('mes.qc.spc.chartTypeXbarR')" value="XBAR_R" />
            <el-option :label="t('mes.qc.spc.chartTypeXbarS')" value="XBAR_S" />
            <el-option :label="t('mes.qc.spc.chartTypeXMr')" value="X_MR" />
            <el-option :label="t('mes.qc.spc.chartTypePChart')" value="P_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeNpChart')" value="NP_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeCChart')" value="C_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeUChart')" value="U_CHART" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.qc.spc.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.qc.spc.statusPlaceholder')" clearable>
            <el-option :label="t('mes.qc.spc.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.spc.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="resetQuery">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.spc.add') }}</el-button>
      </div>

      <el-table :data="chartList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.spc.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.spc.chartCode')" prop="chartCode" width="120" />
        <el-table-column :label="t('mes.qc.spc.chartName')" prop="chartName" width="150" />
        <el-table-column :label="t('mes.qc.spc.chartType')" prop="chartType" width="120">
          <template #default="{ row }">
            {{ getChartTypeText(row.chartType) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.spc.parameterName')" prop="parameterName" width="120" />
        <el-table-column :label="t('mes.qc.spc.unit')" prop="unit" width="80" />
        <el-table-column :label="t('mes.qc.spc.targetValue')" prop="targetValue" width="100" />
        <el-table-column :label="t('mes.qc.spc.ucl')" prop="ucl" width="80" />
        <el-table-column :label="t('mes.qc.spc.lcl')" prop="lcl" width="80" />
        <el-table-column :label="t('mes.qc.spc.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
              {{ row.status === 'ACTIVE' ? t('mes.qc.spc.statusActive') : t('mes.qc.spc.statusInactive') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('common.operation')" fixed="right" width="180">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleEnable(row)" v-if="row.status === 'INACTIVE'">{{ t('mes.qc.spc.enable') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDisable(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.qc.spc.disable') }}</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { SpcControlChart, ChartType } from '@/apis/qc'
import {
  getSpcControlChartListAPI,
  deleteSpcControlChartAPI,
  enableSpcControlChartAPI,
  disableSpcControlChartAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const chartList = ref<SpcControlChart[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  chartCode: '',
  chartName: '',
  chartType: '' as ChartType | '',
  status: ''
})

const getChartTypeText = (type: string) => {
  const textMap: Record<string, string> = {
    XBAR_R: t('mes.qc.spc.chartTypeXbarR'),
    XBAR_S: t('mes.qc.spc.chartTypeXbarS'),
    X_MR: t('mes.qc.spc.chartTypeXMr'),
    P_CHART: t('mes.qc.spc.chartTypePChart'),
    NP_CHART: t('mes.qc.spc.chartTypeNpChart'),
    C_CHART: t('mes.qc.spc.chartTypeCChart'),
    U_CHART: t('mes.qc.spc.chartTypeUChart')
  }
  return textMap[type] || type
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getSpcControlChartListAPI()
    chartList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.spc.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.chartCode = ''
  queryParams.chartName = ''
  queryParams.chartType = ''
  queryParams.status = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/spc/form' })
}

const handleView = (row: SpcControlChart) => {
  router.push({ path: '/mes/spc/detail', query: { id: row.id } })
}

const handleEdit = (row: SpcControlChart) => {
  router.push({ path: '/mes/spc/form', query: { id: row.id } })
}

const handleEnable = async (row: SpcControlChart) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.spc.confirmEnable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await enableSpcControlChartAPI(row.id!)
    ElMessage.success(t('mes.qc.spc.enableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.spc.enableFailed'))
    }
  }
}

const handleDisable = async (row: SpcControlChart) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.spc.confirmDisable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await disableSpcControlChartAPI(row.id!)
    ElMessage.success(t('mes.qc.spc.disableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.spc.disableFailed'))
    }
  }
}

const handleDelete = async (row: SpcControlChart) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.spc.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteSpcControlChartAPI(row.id!)
    ElMessage.success(t('mes.qc.spc.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.spc.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.spc-control-chart-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>