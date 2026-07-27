<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Plus, Trophy } from '@element-plus/icons-vue'
import { ArrowRight } from '@element-plus/icons-vue'
import {
  createActivityAPI,
  getActivitiesAPI,
  participateActivityAPI,
  claimRewardAPI,
  type Activity,
  type CreateActivityRequest,
} from '@/apis/solana'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()
const router = useRouter()

const activities = ref<Activity[]>([])
const listLoading = ref(false)
const createDialogVisible = ref(false)
const dialogLoading = ref(false)

const formData = ref<CreateActivityRequest>({
  authority: '',
  startTime: Math.floor(Date.now() / 1000) + 3600,
  endTime: Math.floor(Date.now() / 1000) + 86400 * 7,
  entryFee: 100,
  rewardPercentages: [5000, 3000, 2000],
  paymentMint: 'So11111111111111111111111111111111111111112',
})

const participateForm = ref({ activityAddress: '', participant: '' })
const participateDialogVisible = ref(false)
const claimForm = ref({ activityAddress: '', winner: '', rank: 1 })
const claimDialogVisible = ref(false)

async function fetchActivities() {
  listLoading.value = true
  try {
    const res = await getActivitiesAPI()
    activities.value = res.data || []
  } catch (e) {
    console.error('fetch activities failed', e)
  } finally {
    listLoading.value = false
  }
}

onMounted(fetchActivities)

function goToDetail(activity: Activity) {
  router.push({ name: 'solanaActivityDetail', query: { activityAddress: activity.activityAddress } })
}

async function handleCreate() {
  if (!formData.value.authority) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    const res = await createActivityAPI(formData.value)
    activities.value.unshift(res.data)
    createDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('create activity failed', e)
  } finally {
    dialogLoading.value = false
  }
}

function showParticipate(activity: Activity) {
  participateForm.value = { activityAddress: activity.activityAddress, participant: '' }
  participateDialogVisible.value = true
}

async function handleParticipate() {
  if (!participateForm.value.participant) return
  dialogLoading.value = true
  try {
    await participateActivityAPI(participateForm.value.activityAddress, participateForm.value.participant)
    participateDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('participate failed', e)
  } finally {
    dialogLoading.value = false
  }
}

function showClaim(activity: Activity) {
  claimForm.value = { activityAddress: activity.activityAddress, winner: '', rank: 1 }
  claimDialogVisible.value = true
}

async function handleClaim() {
  if (!claimForm.value.winner) return
  dialogLoading.value = true
  try {
    await claimRewardAPI(claimForm.value.activityAddress, claimForm.value.winner, claimForm.value.rank)
    claimDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('claim failed', e)
  } finally {
    dialogLoading.value = false
  }
}
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-title">
        <el-icon><Trophy /></el-icon>
        {{ t('solana.activity.title') }}
      </div>
      <el-button type="primary" :icon="Plus" @click="createDialogVisible = true">
        {{ t('solana.activity.create') }}
      </el-button>
    </div>

    <el-card shadow="never">
      <el-table :data="activities" v-loading="listLoading" stripe style="width: 100%">
        <el-table-column label="Activity Address" min-width="180" show-overflow-tooltip>
          <template #default="{ row }">
            <el-link type="primary" :underline="false" @click="goToDetail(row)">
              {{ row.activityAddress }}
              <el-icon style="margin-left: 4px"><ArrowRight /></el-icon>
            </el-link>
          </template>
        </el-table-column>
        <el-table-column prop="authority" label="Authority" min-width="160" show-overflow-tooltip />
        <el-table-column :label="t('solana.activity.entryFee')" width="100">
          <template #default="{ row }">{{ row.entryFee }}</template>
        </el-table-column>
        <el-table-column :label="t('solana.activity.totalPool')" width="100">
          <template #default="{ row }">{{ row.totalPool }}</template>
        </el-table-column>
        <el-table-column :label="t('solana.activity.participants')" width="100">
          <template #default="{ row }">{{ row.participantCount }}</template>
        </el-table-column>
        <el-table-column :label="t('common.createTime')" width="160">
          <template #default="{ row }">{{ formatDateTime(row.startTime * 1000) }}</template>
        </el-table-column>
        <el-table-column :label="t('common.actions')" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="showParticipate(row)">
              {{ t('solana.activity.participate') }}
            </el-button>
            <el-button link type="success" size="small" @click="showClaim(row)">
              {{ t('solana.activity.claim') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="createDialogVisible" :title="t('solana.activity.create')" width="500px">
      <el-form :model="formData" label-width="140px">
        <el-form-item :label="t('solana.activity.authority')" required>
          <el-input v-model="formData.authority" />
        </el-form-item>
        <el-form-item :label="t('solana.activity.entryFee')" required>
          <el-input-number v-model="formData.entryFee" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Start Time">
          <el-input-number v-model="formData.startTime" :min="Math.floor(Date.now() / 1000)" style="width: 100%" />
        </el-form-item>
        <el-form-item label="End Time">
          <el-input-number v-model="formData.endTime" :min="formData.startTime + 1" style="width: 100%" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="participateDialogVisible" :title="t('solana.activity.participate')" width="500px">
      <el-form :model="participateForm" label-width="120px">
        <el-form-item :label="t('solana.activity.participant')" required>
          <el-input v-model="participateForm.participant" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="participateDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleParticipate">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="claimDialogVisible" :title="t('solana.activity.claim')" width="500px">
      <el-form :model="claimForm" label-width="120px">
        <el-form-item :label="t('solana.activity.winner')" required>
          <el-input v-model="claimForm.winner" />
        </el-form-item>
        <el-form-item label="Rank" required>
          <el-radio-group v-model="claimForm.rank">
            <el-radio :value="1">1st</el-radio>
            <el-radio :value="2">2nd</el-radio>
            <el-radio :value="3">3rd</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="claimDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleClaim">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.page-title {
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
