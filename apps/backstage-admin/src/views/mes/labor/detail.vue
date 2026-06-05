<template>
  <div class="app-container">
    <div v-if="loading" v-loading="loading" style="height: 200px" />
    <div v-else-if="operator">
      <el-card class="box-card">
        <template #header>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>{{ t('mes.labor.detail') }} - {{ operator.operatorName }}</span>
            <el-button @click="handleBack">{{ t('common.back') }}</el-button>
          </div>
        </template>
        <el-descriptions :column="3" border>
          <el-descriptions-item :label="t('mes.labor.operatorCode')">{{ operator.operatorCode }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.operatorName')">{{ operator.operatorName }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.status')">
            <el-tag :type="getStatusType(operator.status)">{{ t(`mes.labor.operatorStatus${operator.status}`) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.department')">{{ operator.department }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.jobTitle')">{{ operator.jobTitle || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.shiftGroup')">{{ operator.shiftGroup }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.phone')">{{ operator.phone || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.email')">{{ operator.email || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('mes.labor.hireDate')">{{ operator.hireDate ? new Date(operator.hireDate).toLocaleDateString() : '-' }}</el-descriptions-item>
        </el-descriptions>
      </el-card>

      <el-card class="box-card" style="margin-top: 16px;">
        <template #header><span>{{ t('mes.labor.skills') }}</span></template>
        <el-table v-if="operator.skills && operator.skills.length > 0" :data="operator.skills" border>
          <el-table-column :label="t('mes.labor.skillCode')" prop="skillCode" width="120" />
          <el-table-column :label="t('mes.labor.skillName')" prop="skillName" min-width="120" />
          <el-table-column :label="t('mes.labor.skillLevel')" prop="skillLevel" width="100">
            <template #default="{ row }">
              <el-tag :type="getSkillLevelType(row.skillLevel)">{{ t(`mes.labor.skillLevel${row.skillLevel}`) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.certified')" width="80">
            <template #default="{ row }">
              <el-tag :type="row.certified ? 'success' : 'info'">{{ row.certified ? t('common.yes') : t('common.no') }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.expiryAt')" width="100">
            <template #default="{ row }">{{ row.expiryAt ? new Date(row.expiryAt).toLocaleDateString() : '-' }}</template>
          </el-table-column>
        </el-table>
        <el-empty v-else :description="t('mes.labor.noSkills')" />
      </el-card>
    </div>
    <div v-else><el-empty :description="t('mes.labor.dataNotExist')" /></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { getOperatorByIdAPI } from '@/apis/labor'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const loading = ref(false)
const operator = ref<any>(null)

function getStatusType(s: string): 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { ACTIVE: 'success', INACTIVE: 'info', ON_LEAVE: 'warning', TERMINATED: 'danger' }
  return map[s] || 'info'
}
function getSkillLevelType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { TRAINEE: 'info', JUNIOR: 'primary', MIDDLE: 'warning', SENIOR: 'danger', MASTER: 'success' }
  return map[s] || 'info'
}

async function load(id: number) {
  loading.value = true
  try { const res = await getOperatorByIdAPI(id); operator.value = res.data }
  catch { /* error */ }
  finally { loading.value = false }
}
function handleBack() { router.push({ name: 'labor' }) }

onMounted(() => { if (route.query.id) load(Number(route.query.id)) })
</script>
