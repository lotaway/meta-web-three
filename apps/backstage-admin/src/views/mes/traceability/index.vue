<template>
  <div class="app-container">
    <el-tabs v-model="activeTab">
      <!-- Trace Records Tab -->
      <el-tab-pane :label="t('mes.traceability.recordList')" name="records">
        <div class="filter-container">
          <el-input v-model="listQuery.traceCode" :placeholder="t('mes.traceability.traceCodePlaceholder')"
            class="filter-item" style="width: 160px" @keyup.enter="handleFilter" />
          <el-input v-model="listQuery.batchNo" :placeholder="t('mes.traceability.batchNoPlaceholder')"
            class="filter-item" style="width: 160px" @keyup.enter="handleFilter" />
          <el-input v-model="listQuery.productCode" :placeholder="t('mes.traceability.productCodePlaceholder') || '产品编码'"
            class="filter-item" style="width: 160px" @keyup.enter="handleFilter" />
          <el-input v-model="listQuery.sn" :placeholder="t('mes.traceability.snPlaceholder')"
            class="filter-item" style="width: 160px" @keyup.enter="handleFilter" />
          <el-button class="filter-item" type="primary" :icon="Search" @click="handleFilter">
            {{ t('common.query') }}
          </el-button>
          <el-button class="filter-item" type="primary" :icon="Plus" @click="handleCreate">
            {{ t('mes.traceability.create') }}
          </el-button>
        </div>

        <el-table v-loading="listLoading" :data="recordList" border style="width: 100%">
          <el-table-column type="expand">
            <template #default="{ row }">
              <div v-if="row.relations && row.relations.length > 0" style="padding: 12px;">
                <h4 style="margin-bottom: 8px;">{{ t('mes.traceability.recordList') }} ({{ row.relations.length }})</h4>
                <el-table :data="row.relations" size="small" border>
                  <el-table-column :label="t('mes.traceability.relatedCode')" prop="relatedCode" min-width="120" />
                  <el-table-column :label="t('mes.traceability.traceType')" prop="relatedType" width="100">
                    <template #default="{ row: rel }">
                      {{ t(`mes.traceability.traceType${rel.relatedType}`) }}
                    </template>
                  </el-table-column>
                  <el-table-column :label="t('mes.traceability.relationType')" prop="relationType" width="80">
                    <template #default="{ row: rel }">
                      <el-tag :type="getRelationTagType(rel.relationType)" size="small">
                        {{ t(`mes.traceability.relationType${rel.relationType}`) }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column :label="'Qty'" prop="quantity" width="60" />
                </el-table>
              </div>
              <el-empty v-else :description="t('mes.traceability.noRecords')" />
            </template>
          </el-table-column>
          <el-table-column :label="t('common.id')" prop="id" width="60" />
          <el-table-column :label="t('mes.traceability.traceCode')" prop="traceCode" width="140" />
          <el-table-column :label="t('mes.traceability.traceType')" prop="traceType" width="90">
            <template #default="{ row }">
              {{ t(`mes.traceability.traceType${row.traceType}`) }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.traceability.batchNo')" prop="batchNo" width="120" />
          <el-table-column :label="t('mes.traceability.sn')" prop="sn" width="120" />
          <el-table-column :label="t('mes.traceability.productCode')" prop="productCode" width="120" />
          <el-table-column :label="t('mes.traceability.source')" prop="source" width="100">
            <template #default="{ row }">
              {{ row.source ? t(`mes.traceability.source${row.source}`) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.traceability.createdAt')" width="140">
            <template #default="{ row }">
              {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('common.operations')" width="160" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="handleForwardTrace(row)">
                {{ t('mes.traceability.forwardTrace') }}
              </el-button>
              <el-button type="danger" link @click="handleDelete(row)">
                {{ t('common.delete') }}
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Trace Chain Tab -->
      <el-tab-pane :label="t('mes.traceability.traceChain')" name="chain">
        <div class="filter-container">
          <el-input v-model="chainQuery.traceCode" :placeholder="t('mes.traceability.traceCodePlaceholder')"
            class="filter-item" style="width: 200px" @keyup.enter="handleChainQuery" />
          <el-button class="filter-item" type="primary" :icon="Search" @click="handleChainQuery">
            {{ t('common.query') }}
          </el-button>
          <el-button class="filter-item" type="success" @click="handleForwardChain">
            {{ t('mes.traceability.forwardTrace') }}
          </el-button>
          <el-button class="filter-item" type="warning" @click="handleBackwardChain">
            {{ t('mes.traceability.backwardTrace') }}
          </el-button>
          <el-button class="filter-item" type="primary" @click="handleFullChain">
            {{ t('mes.traceability.fullChain') }}
          </el-button>
        </div>

        <el-alert v-if="chainError" :title="chainError" type="error" show-icon closable
          style="margin-bottom: 16px;" @close="chainError = ''" />

        <div v-loading="chainLoading">
          <!-- Root Node -->
          <el-card v-if="chainResult.root" class="chain-card" shadow="hover">
            <template #header>
              <span>
                <el-tag type="danger" size="small" style="margin-right: 8px;">
                  {{ t('mes.traceability.rootNode') }}
                </el-tag>
                {{ chainResult.root.traceCode }}
              </span>
            </template>
            <el-descriptions :column="3" size="small" border>
              <el-descriptions-item :label="t('mes.traceability.traceType')">
                {{ t(`mes.traceability.traceType${chainResult.root.traceType}`) }}
              </el-descriptions-item>
              <el-descriptions-item :label="t('mes.traceability.batchNo')">
                {{ chainResult.root.batchNo || '-' }}
              </el-descriptions-item>
              <el-descriptions-item :label="t('mes.traceability.productCode')">
                {{ chainResult.root.productCode || '-' }}
              </el-descriptions-item>
              <el-descriptions-item :label="t('mes.traceability.sn')">
                {{ chainResult.root.sn || '-' }}
              </el-descriptions-item>
              <el-descriptions-item :label="t('mes.traceability.source')">
                {{ chainResult.root.source ? t(`mes.traceability.source${chainResult.root.source}`) : '-' }}
              </el-descriptions-item>
              <el-descriptions-item :label="t('mes.traceability.createdAt')">
                {{ chainResult.root.createdAt ? formatDateTime(chainResult.root.createdAt) : '-' }}
              </el-descriptions-item>
            </el-descriptions>
          </el-card>

          <!-- Forward Path -->
          <el-card v-if="chainResult.forwardPath && chainResult.forwardPath.length > 0" class="chain-card" shadow="hover">
            <template #header>
              <span>
                <el-tag type="success" size="small" style="margin-right: 8px;">
                  {{ t('mes.traceability.forwardPath') }}
                </el-tag>
                ({{ chainResult.forwardPath.length }} {{ t('mes.traceability.nodeCount') }})
              </span>
            </template>
            <el-table :data="chainResult.forwardPath" size="small" border style="width: 100%">
              <el-table-column type="index" width="40" />
              <el-table-column :label="t('mes.traceability.traceCode')" prop="traceCode" min-width="120" />
              <el-table-column :label="t('mes.traceability.traceType')" prop="traceType" width="90">
                <template #default="{ row }">
                  <el-tag size="small">{{ t(`mes.traceability.traceType${row.traceType}`) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column :label="t('mes.traceability.batchNo')" prop="batchNo" width="110" />
              <el-table-column :label="t('mes.traceability.productCode')" prop="productCode" width="110" />
              <el-table-column :label="t('mes.traceability.sn')" prop="sn" width="110" />
              <el-table-column :label="t('mes.traceability.createdAt')" width="130">
                <template #default="{ row }">
                  {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
                </template>
              </el-table-column>
            </el-table>
          </el-card>

          <!-- Backward Path -->
          <el-card v-if="chainResult.backwardPath && chainResult.backwardPath.length > 0" class="chain-card" shadow="hover">
            <template #header>
              <span>
                <el-tag type="warning" size="small" style="margin-right: 8px;">
                  {{ t('mes.traceability.backwardPath') }}
                </el-tag>
                ({{ chainResult.backwardPath.length }} {{ t('mes.traceability.nodeCount') }})
              </span>
            </template>
            <el-table :data="chainResult.backwardPath" size="small" border style="width: 100%">
              <el-table-column type="index" width="40" />
              <el-table-column :label="t('mes.traceability.traceCode')" prop="traceCode" min-width="120" />
              <el-table-column :label="t('mes.traceability.traceType')" prop="traceType" width="90">
                <template #default="{ row }">
                  <el-tag size="small">{{ t(`mes.traceability.traceType${row.traceType}`) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column :label="t('mes.traceability.batchNo')" prop="batchNo" width="110" />
              <el-table-column :label="t('mes.traceability.productCode')" prop="productCode" width="110" />
              <el-table-column :label="t('mes.traceability.sn')" prop="sn" width="110" />
              <el-table-column :label="t('mes.traceability.createdAt')" width="130">
                <template #default="{ row }">
                  {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
                </template>
              </el-table-column>
            </el-table>
          </el-card>

          <!-- Forward trace result (linear list from domain service) -->
          <el-card v-if="forwardList && forwardList.length > 0 && !chainResult.root" class="chain-card" shadow="hover">
            <template #header>
              <span>
                <el-tag type="success" size="small" style="margin-right: 8px;">
                  {{ t('mes.traceability.forwardTrace') }}
                </el-tag>
                ({{ forwardList.length }} {{ t('mes.traceability.nodeCount') }})
              </span>
            </template>
            <el-table :data="forwardList" size="small" border style="width: 100%">
              <el-table-column type="index" width="40" />
              <el-table-column :label="t('mes.traceability.traceCode')" prop="traceCode" min-width="120" />
              <el-table-column :label="t('mes.traceability.traceType')" prop="traceType" width="90">
                <template #default="{ row }">
                  <el-tag size="small">{{ t(`mes.traceability.traceType${row.traceType}`) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column :label="t('mes.traceability.batchNo')" prop="batchNo" width="110" />
              <el-table-column :label="t('mes.traceability.productCode')" prop="productCode" width="110" />
              <el-table-column :label="t('mes.traceability.createdAt')" width="130">
                <template #default="{ row }">
                  {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
                </template>
              </el-table-column>
            </el-table>
          </el-card>

          <!-- Backward trace result -->
          <el-card v-if="backwardList && backwardList.length > 0 && !chainResult.root" class="chain-card" shadow="hover">
            <template #header>
              <span>
                <el-tag type="warning" size="small" style="margin-right: 8px;">
                  {{ t('mes.traceability.backwardTrace') }}
                </el-tag>
                ({{ backwardList.length }} {{ t('mes.traceability.nodeCount') }})
              </span>
            </template>
            <el-table :data="backwardList" size="small" border style="width: 100%">
              <el-table-column type="index" width="40" />
              <el-table-column :label="t('mes.traceability.traceCode')" prop="traceCode" min-width="120" />
              <el-table-column :label="t('mes.traceability.traceType')" prop="traceType" width="90">
                <template #default="{ row }">
                  <el-tag size="small">{{ t(`mes.traceability.traceType${row.traceType}`) }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column :label="t('mes.traceability.batchNo')" prop="batchNo" width="110" />
              <el-table-column :label="t('mes.traceability.productCode')" prop="productCode" width="110" />
              <el-table-column :label="t('mes.traceability.createdAt')" width="130">
                <template #default="{ row }">
                  {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
                </template>
              </el-table-column>
            </el-table>
          </el-card>

          <el-empty v-if="!chainResult.root && !forwardList.length && !backwardList.length && !chainLoading"
            :description="t('mes.traceability.noRecords')" />
        </div>
      </el-tab-pane>
    </el-tabs>

    <!-- Create Dialog -->
    <el-dialog :title="t('mes.traceability.create')" v-model="createDialogVisible" width="550px">
      <el-form :model="createForm" label-width="110px">
        <el-form-item :label="t('mes.traceability.traceCode')" :required="true">
          <el-input v-model="createForm.traceCode" />
        </el-form-item>
        <el-row :gutter="12">
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.traceType')" :required="true">
              <el-select v-model="createForm.traceType" style="width: 100%">
                <el-option v-for="tt in traceTypeOptions" :key="tt.value" :label="tt.label" :value="tt.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.source')">
              <el-select v-model="createForm.source" style="width: 100%" clearable>
                <el-option v-for="s in sourceOptions" :key="s.value" :label="s.label" :value="s.value" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.productCode')">
              <el-input v-model="createForm.productCode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.batchNo')">
              <el-input v-model="createForm.batchNo" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="12">
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.sn')">
              <el-input v-model="createForm.sn" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.traceability.workOrderNo')">
              <el-input v-model="createForm.workOrderNo" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="createSubmitting" @click="submitCreate">
          {{ t('common.confirm') }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { Search, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  getTraceRecordListAPI,
  createTraceRecordAPI,
  deleteTraceRecordAPI,
  forwardTraceAPI,
  backwardTraceAPI,
  getTraceChainAPI,
  type TraceRecord,
  type TraceChain,
  type TraceType,
  type TraceSource,
} from '@/apis/traceability-mes'

const { t } = useI18n()

const activeTab = ref('records')
const listLoading = ref(false)
const chainLoading = ref(false)

const listQuery = ref({
  traceCode: '',
  batchNo: '',
  productCode: '',
  sn: '',
})

const recordList = ref<TraceRecord[]>([])

const chainQuery = ref({ traceCode: '' })
const chainResult = ref<TraceChain>({ root: null as unknown as TraceRecord, forwardPath: [], backwardPath: [] })
const forwardList = ref<TraceRecord[]>([])
const backwardList = ref<TraceRecord[]>([])
const chainError = ref('')

const createDialogVisible = ref(false)
const createSubmitting = ref(false)
const createForm = ref({
  traceCode: '',
  traceType: 'PRODUCT' as TraceType,
  productCode: '',
  productName: '',
  batchNo: '',
  sn: '',
  source: '' as TraceSource | '',
  workOrderNo: '',
})

const traceTypeOptions = [
  { value: 'PRODUCT', label: t('mes.traceability.traceTypePRODUCT') },
  { value: 'BATCH', label: t('mes.traceability.traceTypeBATCH') },
  { value: 'MATERIAL', label: t('mes.traceability.traceTypeMATERIAL') },
  { value: 'SN', label: t('mes.traceability.traceTypeSN') },
  { value: 'WORK_ORDER', label: t('mes.traceability.traceTypeWORK_ORDER') },
  { value: 'PROCESS', label: t('mes.traceability.traceTypePROCESS') },
  { value: 'QC', label: t('mes.traceability.traceTypeQC') },
  { value: 'EQUIPMENT', label: t('mes.traceability.traceTypeEQUIPMENT') },
  { value: 'OPERATOR', label: t('mes.traceability.traceTypeOPERATOR') },
]

const sourceOptions = [
  { value: 'WORK_ORDER', label: t('mes.traceability.sourceWORK_ORDER') },
  { value: 'PRODUCTION_TASK', label: t('mes.traceability.sourcePRODUCTION_TASK') },
  { value: 'MATERIAL_ISSUE', label: t('mes.traceability.sourceMATERIAL_ISSUE') },
  { value: 'QC_INSPECTION', label: t('mes.traceability.sourceQC_INSPECTION') },
  { value: 'EQUIPMENT', label: t('mes.traceability.sourceEQUIPMENT') },
  { value: 'ANDON', label: t('mes.traceability.sourceANDON') },
]

function getRelationTagType(rt: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    CONSUMED: 'warning', SPLIT_FROM: 'primary', PARENT: 'info', BIND: 'success',
  }
  return map[rt] || 'info'
}

function formatDateTime(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

async function getList() {
  listLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (listQuery.value.batchNo) params.batchNo = listQuery.value.batchNo
    if (listQuery.value.productCode) params.productCode = listQuery.value.productCode
    if (listQuery.value.sn) params.sn = listQuery.value.sn
    const res = await getTraceRecordListAPI(params)
    recordList.value = res.data || []
  } catch {
    ElMessage.error(t('mes.traceability.loadFailed'))
  } finally {
    listLoading.value = false
  }
}

function handleFilter() {
  getList()
}

function handleCreate() {
  createForm.value = {
    traceCode: '',
    traceType: 'PRODUCT',
    productCode: '',
    productName: '',
    batchNo: '',
    sn: '',
    source: '',
    workOrderNo: '',
  }
  createDialogVisible.value = true
}

async function submitCreate() {
  if (!createForm.value.traceCode) {
    ElMessage.warning('Please enter trace code')
    return
  }
  createSubmitting.value = true
  try {
    const data: Record<string, unknown> = {
      traceCode: createForm.value.traceCode,
      traceType: createForm.value.traceType,
      productCode: createForm.value.productCode || undefined,
      productName: createForm.value.productName || undefined,
      batchNo: createForm.value.batchNo || undefined,
      sn: createForm.value.sn || undefined,
      source: createForm.value.source || undefined,
      workOrderNo: createForm.value.workOrderNo || undefined,
    }
    await createTraceRecordAPI(data as any)
    ElMessage.success(t('mes.traceability.createSuccess'))
    createDialogVisible.value = false
    getList()
  } catch {
    ElMessage.error(t('common.submitFailed'))
  } finally {
    createSubmitting.value = false
  }
}

async function handleDelete(row: TraceRecord) {
  try {
    await ElMessageBox.confirm(t('mes.traceability.deleteConfirm'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await deleteTraceRecordAPI(row.id!)
    ElMessage.success(t('mes.traceability.deleteSuccess'))
    getList()
  } catch { /* cancel */ }
}

async function handleForwardTrace(row: TraceRecord) {
  activeTab.value = 'chain'
  chainQuery.value.traceCode = row.traceCode
  await handleForwardChain()
}

async function handleChainQuery() {
  chainError.value = ''
  forwardList.value = []
  backwardList.value = []
  chainResult.value = { root: null as unknown as TraceRecord, forwardPath: [], backwardPath: [] }
  if (!chainQuery.value.traceCode) {
    chainError.value = 'Please enter a trace code'
    return
  }
  chainLoading.value = true
  try {
    const res = await getTraceChainAPI(chainQuery.value.traceCode)
    chainResult.value = res.data
  } catch {
    chainError.value = 'Trace code not found or query failed'
  } finally {
    chainLoading.value = false
  }
}

async function handleForwardChain() {
  chainError.value = ''
  forwardList.value = []
  chainResult.value = { root: null as unknown as TraceRecord, forwardPath: [], backwardPath: [] }
  if (!chainQuery.value.traceCode) {
    chainError.value = 'Please enter a trace code'
    return
  }
  chainLoading.value = true
  try {
    const res = await forwardTraceAPI(chainQuery.value.traceCode)
    forwardList.value = res.data || []
  } catch {
    chainError.value = 'Forward trace query failed'
  } finally {
    chainLoading.value = false
  }
}

async function handleBackwardChain() {
  chainError.value = ''
  backwardList.value = []
  chainResult.value = { root: null as unknown as TraceRecord, forwardPath: [], backwardPath: [] }
  if (!chainQuery.value.traceCode) {
    chainError.value = 'Please enter a trace code'
    return
  }
  chainLoading.value = true
  try {
    const res = await backwardTraceAPI(chainQuery.value.traceCode)
    backwardList.value = res.data || []
  } catch {
    chainError.value = 'Backward trace query failed'
  } finally {
    chainLoading.value = false
  }
}

async function handleFullChain() {
  await handleChainQuery()
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.filter-container {
  padding-bottom: 16px;
}
.filter-item {
  margin-right: 8px;
  vertical-align: top;
}
.chain-card {
  margin-bottom: 16px;
}
</style>
