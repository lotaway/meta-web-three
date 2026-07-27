<template>
  <div class="activity-detail-container">
    <el-card v-loading="loading">
      <template #header>
        <div class="header-row">
          <span>{{ t('solana.activity.detail') }}</span>
          <el-button @click="handleBack">{{ t('solana.activity.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-if="activity">
        <el-descriptions-item :label="t('solana.activity.activityAddress')" :span="2">
          {{ activity.activityAddress }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.authority')" :span="2">
          {{ activity.authority }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.startTime')">
          {{ formatDateTime(activity.startTime * 1000) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.endTime')">
          {{ formatDateTime(activity.endTime * 1000) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.entryFee')">
          {{ activity.entryFee }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.totalPool')">
          {{ activity.totalPool }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.participants')">
          {{ activity.participantCount }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.rewardPercentages')">
          {{ activity.rewardPercentages.join('%, ') }}%
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.activity.txSignature')" :span="2">
          {{ activity.txSignature }}
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />

      <div class="action-buttons">
        <el-button type="primary" @click="handleParticipate">
          {{ t('solana.activity.participateNow') }}
        </el-button>
        <el-button type="success" @click="handleClaim">
          {{ t('solana.activity.claimNow') }}
        </el-button>
      </div>
    </el-card>

    <el-dialog v-model="participateDialogVisible" :title="t('solana.activity.participate')" width="500px">
      <el-form :model="participateForm" label-width="120px">
        <el-form-item :label="t('solana.activity.participant')" required>
          <el-input v-model="participateForm.participant" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="participateDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="submitting" @click="handleParticipateSubmit">{{ t('common.confirm') }}</el-button>
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
        <el-button type="primary" :loading="submitting" @click="handleClaimSubmit">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { getActivityAPI, participateActivityAPI, claimRewardAPI, type Activity } from '@/apis/solana'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const submitting = ref(false)
const activity = ref<Activity | null>(null)

const participateDialogVisible = ref(false)
const participateForm = ref({ participant: '' })

const claimDialogVisible = ref(false)
const claimForm = ref({ winner: '', rank: 1 })

async function fetchDetail() {
  const activityAddress = route.query.activityAddress as string
  if (!activityAddress) {
    ElMessage.warning('Missing activity address')
    return
  }
  loading.value = true
  try {
    const res = await getActivityAPI(activityAddress)
    activity.value = res.data
  } catch (e) {
    console.error('fetch activity detail failed', e)
  } finally {
    loading.value = false
  }
}

onMounted(fetchDetail)

function handleBack() {
  router.push({ name: 'solanaActivity' })
}

function handleParticipate() {
  participateForm.value.participant = ''
  participateDialogVisible.value = true
}

async function handleParticipateSubmit() {
  if (!participateForm.value.participant) return
  if (!activity.value) return
  submitting.value = true
  try {
    await participateActivityAPI(activity.value.activityAddress, participateForm.value.participant)
    participateDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
    fetchDetail()
  } catch (e) {
    console.error('participate failed', e)
  } finally {
    submitting.value = false
  }
}

function handleClaim() {
  claimForm.value = { winner: '', rank: 1 }
  claimDialogVisible.value = true
}

async function handleClaimSubmit() {
  if (!claimForm.value.winner) return
  if (!activity.value) return
  submitting.value = true
  try {
    await claimRewardAPI(activity.value.activityAddress, claimForm.value.winner, claimForm.value.rank)
    claimDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('claim failed', e)
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.activity-detail-container {
  max-width: 900px;
  margin: 0 auto;
}
.header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 16px;
  font-weight: 600;
}
.action-buttons {
  display: flex;
  gap: 12px;
}
</style>
