<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete } from '@element-plus/icons-vue'
import {
  getCouponListAPI,
  couponCreateAPI,
  couponUpdateByIdAPI,
  couponDeleteByIdAPI,
  getCouponHistoryListAPI
} from '@/apis/promotion'
import type { SmsCoupon, CouponQueryParam } from '@/types/coupon'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

const activeTab = ref('coupon')

// Coupon state
const couponListQuery = ref<CouponQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE
})
const couponList = ref<SmsCoupon[]>([])
const couponTotal = ref(0)
const couponLoading = ref(false)
const couponDialogVisible = ref(false)
const couponDialogTitle = ref('')
const couponForm = reactive<SmsCoupon>({
  type: 0,
  name: '',
  platform: 0,
  count: 0,
  amount: 0,
  perLimit: 1,
  minPoint: 0,
  startTime: '',
  endTime: '',
  useType: 0,
  note: '',
  publishCount: 0
})
const couponFormRef = ref()

const getCouponList = async () => {
  couponLoading.value = true
  try {
    const response = await getCouponListAPI(couponListQuery.value)
    couponList.value = response.data.list
    couponTotal.value = response.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    couponLoading.value = false
  }
}

const handleCouponSearch = () => {
  couponListQuery.value.pageNum = 1
  getCouponList()
}

const handleCouponReset = () => {
  couponListQuery.value = {
    pageNum: 1,
    pageSize: DEFAULT_PAGE_SIZE
  }
  getCouponList()
}

const handleCouponAdd = () => {
  couponDialogTitle.value = t('promotion.addCoupon')
  Object.assign(couponForm, {
    id: undefined,
    type: 0,
    name: '',
    platform: 0,
    count: 0,
    amount: 0,
    perLimit: 1,
    minPoint: 0,
    startTime: '',
    endTime: '',
    useType: 0,
    note: '',
    publishCount: 0
  })
  couponDialogVisible.value = true
}

const handleCouponEdit = (row: SmsCoupon) => {
  couponDialogTitle.value = t('promotion.editCoupon')
  Object.assign(couponForm, { ...row })
  couponDialogVisible.value = true
}

const handleCouponDelete = async (row: SmsCoupon) => {
  try {
    await ElMessageBox.confirm(t('promotion.confirmDelete'), t('common.warning'), {
      type: 'warning'
    })
    await couponDeleteByIdAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    getCouponList()
  } catch {
    // cancelled
  }
}

const handleCouponSubmit = async () => {
  try {
    if (couponForm.id) {
      await couponUpdateByIdAPI(couponForm.id, couponForm)
      ElMessage.success(t('common.updateSuccess'))
    } else {
      await couponCreateAPI(couponForm)
      ElMessage.success(t('common.createSuccess'))
    }
    couponDialogVisible.value = false
    getCouponList()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleCouponSizeChange = (val: number) => {
  couponListQuery.value.pageSize = val
  getCouponList()
}

const handleCouponCurrentChange = (val: number) => {
  couponListQuery.value.pageNum = val
  getCouponList()
}

// Coupon type options
const couponTypeOptions = [
  { value: 0, label: t('promotion.couponType0') },
  { value: 1, label: t('promotion.couponType1') },
  { value: 2, label: t('promotion.couponType2') },
  { value: 3, label: t('promotion.couponType3') }
]

const platformOptions = [
  { value: 0, label: t('promotion.platformAll') },
  { value: 1, label: t('promotion.platformMobile') },
  { value: 2, label: t('promotion.platformPC') }
]

const useTypeOptions = [
  { value: 0, label: t('promotion.useTypeAll') },
  { value: 1, label: t('promotion.useTypeCategory') },
  { value: 2, label: t('promotion.useTypeProduct') }
]

onMounted(() => {
  getCouponList()
})
</script>

<template>
  <div class="promotion-container">
    <el-tabs v-model="activeTab" type="border-card">
      <el-tab-pane :label="t('promotion.couponManagement')" name="coupon">
        <div class="toolbar">
          <el-form :inline="true" :model="couponListQuery">
            <el-form-item :label="t('promotion.couponName')">
              <el-input v-model="couponListQuery.name" :placeholder="t('promotion.couponNamePlaceholder')" clearable />
            </el-form-item>
            <el-form-item :label="t('promotion.couponType')">
              <el-select v-model="couponListQuery.type" :placeholder="t('common.select')" clearable>
                <el-option v-for="item in couponTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleCouponSearch">{{ t('common.search') }}</el-button>
              <el-button @click="handleCouponReset">{{ t('common.reset') }}</el-button>
              <el-button type="primary" :icon="Plus" @click="handleCouponAdd">{{ t('common.add') }}</el-button>
            </el-form-item>
          </el-form>
        </div>

        <el-table v-loading="couponLoading" :data="couponList" border stripe>
          <el-table-column :label="t('promotion.id')" prop="id" width="80" />
          <el-table-column :label="t('promotion.couponName')" prop="name" min-width="150" />
          <el-table-column :label="t('promotion.couponType')" prop="type" width="120">
            <template #default="{ row }">
              {{ couponTypeOptions.find(o => o.value === row.type)?.label || '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('promotion.amount')" prop="amount" width="100">
            <template #default="{ row }">
              ¥{{ row.amount }}
            </template>
          </el-table-column>
          <el-table-column :label="t('promotion.minPoint')" prop="minPoint" width="100">
            <template #default="{ row }">
              {{ row.minPoint === 0 ? t('promotion.noLimit') : `¥${row.minPoint}` }}
            </template>
          </el-table-column>
          <el-table-column :label="t('promotion.publishCount')" prop="publishCount" width="100" />
          <el-table-column :label="t('promotion.useCount')" prop="useCount" width="100" />
          <el-table-column :label="t('promotion.startTime')" prop="startTime" width="160" />
          <el-table-column :label="t('promotion.endTime')" prop="endTime" width="160" />
          <el-table-column :label="t('common.action')" width="150" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link :icon="Edit" @click="handleCouponEdit(row)">{{ t('common.edit') }}</el-button>
              <el-button type="danger" link :icon="Delete" @click="handleCouponDelete(row)">{{ t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>

        <el-pagination
          v-model:current-page="couponListQuery.pageNum"
          v-model:page-size="couponListQuery.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="couponTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleCouponSizeChange"
          @current-change="handleCouponCurrentChange"
        />
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="couponDialogVisible" :title="couponDialogTitle" width="600px">
      <el-form ref="couponFormRef" :model="couponForm" label-width="120px">
        <el-form-item :label="t('promotion.couponName')" prop="name" :rules="[{ required: true, message: t('promotion.nameRequired'), trigger: 'blur' }]">
          <el-input v-model="couponForm.name" :placeholder="t('promotion.couponNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('promotion.couponType')" prop="type">
          <el-select v-model="couponForm.type">
            <el-option v-for="item in couponTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('promotion.platform')" prop="platform">
          <el-select v-model="couponForm.platform">
            <el-option v-for="item in platformOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('promotion.amount')" prop="amount" :rules="[{ required: true, message: t('promotion.amountRequired'), trigger: 'blur' }]">
          <el-input-number v-model="couponForm.amount" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('promotion.minPoint')" prop="minPoint">
          <el-input-number v-model="couponForm.minPoint" :min="0" />
        </el-form-item>
        <el-form-item :label="t('promotion.perLimit')" prop="perLimit">
          <el-input-number v-model="couponForm.perLimit" :min="1" />
        </el-form-item>
        <el-form-item :label="t('promotion.publishCount')" prop="publishCount">
          <el-input-number v-model="couponForm.publishCount" :min="0" />
        </el-form-item>
        <el-form-item :label="t('promotion.useType')" prop="useType">
          <el-select v-model="couponForm.useType">
            <el-option v-for="item in useTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('promotion.startTime')" prop="startTime">
          <el-date-picker v-model="couponForm.startTime" type="datetime" :placeholder="t('promotion.selectTime')" value-format="YYYY-MM-DD HH:mm:ss" />
        </el-form-item>
        <el-form-item :label="t('promotion.endTime')" prop="endTime">
          <el-date-picker v-model="couponForm.endTime" type="datetime" :placeholder="t('promotion.selectTime')" value-format="YYYY-MM-DD HH:mm:ss" />
        </el-form-item>
        <el-form-item :label="t('promotion.note')" prop="note">
          <el-input v-model="couponForm.note" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="couponDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleCouponSubmit">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.promotion-container {
  padding: 20px;
}

.toolbar {
  margin-bottom: 20px;
}

.el-table {
  margin-top: 20px;
}

.el-pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>