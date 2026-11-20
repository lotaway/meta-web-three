<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Tickets } from '@element-plus/icons-vue'
import { getFlashSessionSelectListAPI } from '@/apis/flashSession'
import type { SmsFlashPromotionSession } from '@/types/flash'
import { formatTime } from '@/utils/datetime'

// Get route
const router = useRouter()
const route = useRoute()

// Flash sale time slots
const list = ref<SmsFlashPromotionSession[]>([])
// Loading state
const listLoading = ref(false)

// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const res = await getFlashSessionSelectListAPI({ flashPromotionId: Number(route.query.flashPromotionId) })
    listLoading.value = false
    list.value = res.data
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load flash session list:', error)
  }
}

// Fetch data on mount
onMounted(() => {
  getList()
})

// Show associated products
const handleShowRelation = (index: number, row: SmsFlashPromotionSession) => {
  router.push({
    path: '/sms/flashProductRelation',
    query: {
      flashPromotionId: route.query.flashPromotionId,
      flashPromotionSessionId: row.id
    }
  })
}
</script>

<template>
  <div class="app-container">
    <el-card shadow="never" class="operate-container">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
    </el-card>
    <div class="table-container">
      <el-table ref="selectSessionTable" :data="list" style="width: 100%;" v-loading="listLoading" border>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="秒杀时间段名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="每日开始时间" align="center">
          <template #default="scope">{{ formatTime(scope.row.startTime) }}</template>
        </el-table-column>
        <el-table-column label="每日结束时间" align="center">
          <template #default="scope">{{ formatTime(scope.row.endTime) }}</template>
        </el-table-column>
        <el-table-column label="商品数量" align="center">
          <template #default="scope">{{ scope.row.productCount }}</template>
        </el-table-column>
        <el-table-column label="操作" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleShowRelation(scope.$index, scope.row as SmsFlashPromotionSession)">{{ t('product.productList') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.operate-container {
  margin-top: 0;
}
</style>
