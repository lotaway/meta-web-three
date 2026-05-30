<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { classifyInventoryAPI, type AbcClassification } from '@/apis/inventory'
import { t } from '@/locales'

const list = ref<AbcClassification[]>([])
const listLoading = ref(true)
const periodDays = ref(30)
const warehouseId = ref<number | undefined>(undefined)

const categoryStats = computed(() => {
  const stats = { A: 0, B: 0, C: 0 }
  list.value.forEach(item => {
    stats[item.category]++
  })
  return stats
})

const totalValue = computed(() => {
  return list.value.reduce((sum, item) => sum + (item.totalValue || 0), 0)
})

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, any> = {
      periodDays: periodDays.value
    }
    if (warehouseId.value) {
      params.warehouseId = warehouseId.value
    }
    const response = await classifyInventoryAPI(params)
    listLoading.value = false
    list.value = response.data || []
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load ABC classification:', error)
  }
}

const handleSearch = () => {
  getList()
}

const getCategoryTagType = (category: string) => {
  switch (category) {
    case 'A':
      return 'danger'
    case 'B':
      return 'warning'
    case 'C':
      return 'info'
    default:
      return 'info'
  }
}

const formatValue = (value: number) => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: 'CNY'
  }).format(value || 0)
}

const formatRate = (rate: number) => {
  if (rate === null || rate === undefined) return '-'
  return (rate * 100).toFixed(2) + '%'
}

onMounted(() => {
  getList()
})
</script>

<template>
  <div class="abc-classification-container">
    <div class="filter-container">
      <el-form :inline="true" class="filter-form">
        <el-form-item :label="t('abc.periodDays')">
          <el-input-number
            v-model="periodDays"
            :min="1"
            :max="365"
            :step="7"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">
            {{ t('common.query') }}
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-row :gutter="20" class="stats-row">
      <el-col :span="8">
        <el-card class="stats-card stats-card-a">
          <div class="stats-title">{{ t('abc.categoryA') }}</div>
          <div class="stats-value">{{ categoryStats.A }}</div>
          <div class="stats-desc">{{ t('abc.highValue') }}</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="stats-card stats-card-b">
          <div class="stats-title">{{ t('abc.categoryB') }}</div>
          <div class="stats-value">{{ categoryStats.B }}</div>
          <div class="stats-desc">{{ t('abc.mediumValue') }}</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="stats-card stats-card-c">
          <div class="stats-title">{{ t('abc.categoryC') }}</div>
          <div class="stats-value">{{ categoryStats.C }}</div>
          <div class="stats-desc">{{ t('abc.lowValue') }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-table
      v-loading="listLoading"
      :data="list"
      border
      style="width: 100%"
    >
      <el-table-column prop="rank" :label="t('abc.rank')" width="80" align="center" />
      <el-table-column prop="skuCode" :label="t('abc.skuCode')" min-width="150" />
      <el-table-column :label="t('abc.category')" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="getCategoryTagType(row.category)">
            {{ row.category }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('abc.totalValue')" width="150" align="right">
        <template #default="{ row }">
          {{ formatValue(row.totalValue) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('abc.turnoverRate')" width="150" align="right">
        <template #default="{ row }">
          {{ formatRate(row.turnoverRate) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('abc.strategy')" min-width="200">
        <template #default="{ row }">
          <span v-if="row.category === 'A'">{{ t('abc.strategyA') }}</span>
          <span v-else-if="row.category === 'B'">{{ t('abc.strategyB') }}</span>
          <span v-else>{{ t('abc.strategyC') }}</span>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<style scoped lang="scss">
.abc-classification-container {
  padding: 20px;
}

.filter-container {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  align-items: center;
  gap: 10px;
}

.stats-row {
  margin-bottom: 20px;
}

.stats-card {
  text-align: center;
  
  .stats-title {
    font-size: 14px;
    color: #666;
    margin-bottom: 10px;
  }
  
  .stats-value {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 10px;
  }
  
  .stats-desc {
    font-size: 12px;
    color: #999;
  }
}

.stats-card-a {
  .stats-value {
    color: #f56c6c;
  }
}

.stats-card-b {
  .stats-value {
    color: #e6a23c;
  }
}

.stats-card-c {
  .stats-value {
    color: #909399;
  }
}
</style>