<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Link } from '@element-plus/icons-vue'
import {
  setUplineAPI,
  getCommissionAPI,
  type Commission,
} from '@/apis/solana'

const { t } = useI18n()

const commissionInfo = ref<Commission | null>(null)
const searchAccount = ref('')
const loading = ref(false)

const setUplineForm = ref({ target: '', upline: '' })
const setUplineDialogVisible = ref(false)
const dialogLoading = ref(false)

async function handleSearch() {
  if (!searchAccount.value) return
  loading.value = true
  try {
    const res = await getCommissionAPI(searchAccount.value)
    commissionInfo.value = res.data
  } catch (e) {
    console.error('query commission failed', e)
  } finally {
    loading.value = false
  }
}

async function handleSetUpline() {
  if (!setUplineForm.value.target || !setUplineForm.value.upline) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    const res = await setUplineAPI(setUplineForm.value.target, setUplineForm.value.upline)
    commissionInfo.value = res.data
    setUplineDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('set upline failed', e)
  } finally {
    dialogLoading.value = false
  }
}
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-title">
        <el-icon><Link /></el-icon>
        {{ t('solana.commission.title') }}
      </div>
      <el-button type="primary" :icon="Link" @click="setUplineDialogVisible = true">
        {{ t('solana.commission.setUpline') }}
      </el-button>
    </div>

    <el-card class="search-card" shadow="never">
      <el-form :inline="true">
        <el-form-item :label="t('solana.commission.account')">
          <el-input v-model="searchAccount" :placeholder="t('solana.commission.accountPlaceholder')" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="handleSearch">
            {{ t('common.search') }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" v-if="commissionInfo">
      <el-descriptions :column="1" border>
        <el-descriptions-item :label="t('solana.commission.account')">{{ commissionInfo.account }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.commission.upline')">{{ commissionInfo.upline || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.commission.level')">{{ commissionInfo.level }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.commission.downlineCount')">{{ commissionInfo.downlineCount }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-dialog v-model="setUplineDialogVisible" :title="t('solana.commission.setUpline')" width="500px">
      <el-form :model="setUplineForm" label-width="120px">
        <el-form-item :label="t('solana.commission.target')" required>
          <el-input v-model="setUplineForm.target" />
        </el-form-item>
        <el-form-item :label="t('solana.commission.upline')" required>
          <el-input v-model="setUplineForm.upline" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="setUplineDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSetUpline">{{ t('common.confirm') }}</el-button>
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
.search-card {
  margin-bottom: 16px;
}
</style>
