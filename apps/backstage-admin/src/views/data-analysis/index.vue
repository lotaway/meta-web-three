<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { Search, Refresh } from '@element-plus/icons-vue'
import {
  getInventoryOverviewAPI,
  getLowStockProductsAPI,
  getOverstockProductsAPI,
  getCategorySalesAPI,
  type InventoryOverviewDTO,
  type InventoryAnalysisDTO,
  type CategorySalesDTO
} from '@/apis/dataAnalysis'
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'

const loading = ref(false)
const statistics = ref<InventoryOverviewDTO>({
  totalProducts: 0,
  totalQuantity: 0,
  lowStockCount: 0,
  overstockCount: 0,
  totalValue: 0
})

const query = reactive({
  startDate: '',
  endDate: ''
})

const lowStockList = ref<InventoryAnalysisDTO[]>([])
const overstockList = ref<InventoryAnalysisDTO[]>([])
const categorySalesList = ref<CategorySalesDTO[]>([])
const activeTab = ref('inventory')

const getStatistics = async () => {
  try {
    const res = await getInventoryOverviewAPI()
    if (res.data) {
      statistics.value = res.data
    }
  } catch (error) {
    ElMessage.error('Failed to load inventory overview')
  }
}

const getLowStockList = async () => {
  loading.value = true
  try {
    const res = await getLowStockProductsAPI()
    lowStockList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load low stock products')
  } finally {
    loading.value = false
  }
}

const getOverstockList = async () => {
  loading.value = true
  try {
    const res = await getOverstockProductsAPI()
    overstockList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load overstock products')
  } finally {
    loading.value = false
  }
}

const getCategorySales = async () => {
  loading.value = true
  try {
    const endDateStr = query.endDate || new Date().toISOString().split('T')[0]
    const res = await getCategorySalesAPI(
      query.startDate || '2024-01-01',
      endDateStr as string
    )
    categorySalesList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load category sales')
  } finally {
    loading.value = false
  }
}

const handleRefresh = () => {
  if (activeTab.value === 'inventory') {
    getStatistics()
    getLowStockList()
    getOverstockList()
  } else if (activeTab.value === 'lowStock') {
    getLowStockList()
  } else if (activeTab.value === 'overstock') {
    getOverstockList()
  } else if (activeTab.value === 'sales') {
    getCategorySales()
  }
}

import type { TabPaneName } from 'element-plus'

const handleTabChange = (tab: TabPaneName) => {
  const tabStr = String(tab)
  activeTab.value = tabStr
  if (tabStr === 'inventory') {
    getStatistics()
    getLowStockList()
    getOverstockList()
  } else if (tab === 'lowStock') {
    getLowStockList()
  } else if (tab === 'overstock') {
    getOverstockList()
  } else if (tab === 'sales') {
    getCategorySales()
  }
}

onMounted(() => {
  getStatistics()
  getLowStockList()
  getOverstockList()
})
</script>

<template>
  <div class="data-analysis-container">
    <el-row :gutter="20" class="statistics-row">
      <el-col :span="8">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label">Total Products</div>
            <div class="statistics-value">{{ statistics.totalProducts }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label">Total Quantity</div>
            <div class="statistics-value">{{ statistics.totalQuantity }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label warning">Low Stock</div>
            <div class="statistics-value warning">{{ statistics.lowStockCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label danger">Overstock</div>
            <div class="statistics-value danger">{{ statistics.overstockCount }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="query-card">
      <el-form :inline="true" :model="query" class="query-form">
        <el-form-item label="Date Range">
          <el-date-picker
            v-model="query.startDate"
            type="date"
            placeholder="Start Date"
            value-format="YYYY-MM-DD"
            style="width: 150px"
          />
          <span style="margin: 0 10px">-</span>
          <el-date-picker
            v-model="query.endDate"
            type="date"
            placeholder="End Date"
            value-format="YYYY-MM-DD"
            style="width: 150px"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="getCategorySales">Search</el-button>
          <el-button :icon="Refresh" @click="handleRefresh">Refresh</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="table-card">
      <el-tabs v-model="activeTab" @tab-change="handleTabChange">
        <el-tab-pane label="Inventory Overview" name="inventory">
          <el-row :gutter="20">
            <el-col :span="12">
              <div class="section-title">Low Stock Products</div>
              <el-table v-loading="loading" :data="lowStockList" stripe max-height="400">
                <el-table-column prop="productId" label="Product ID" width="120" />
                <el-table-column prop="productName" label="Product Name" min-width="150" show-overflow-tooltip />
                <el-table-column prop="quantity" label="Quantity" width="100" />
                <el-table-column prop="minStock" label="Min Stock" width="100" />
                <el-table-column prop="warehouseName" label="Warehouse" width="120" />
              </el-table>
            </el-col>
            <el-col :span="12">
              <div class="section-title">Overstock Products</div>
              <el-table v-loading="loading" :data="overstockList" stripe max-height="400">
                <el-table-column prop="productId" label="Product ID" width="120" />
                <el-table-column prop="productName" label="Product Name" min-width="150" show-overflow-tooltip />
                <el-table-column prop="quantity" label="Quantity" width="100" />
                <el-table-column prop="maxStock" label="Max Stock" width="100" />
                <el-table-column prop="warehouseName" label="Warehouse" width="120" />
              </el-table>
            </el-col>
          </el-row>
        </el-tab-pane>

        <el-tab-pane label="Low Stock" name="lowStock">
          <el-table v-loading="loading" :data="lowStockList" stripe>
            <el-table-column prop="productId" label="Product ID" width="120" />
            <el-table-column prop="productName" label="Product Name" min-width="150" show-overflow-tooltip />
            <el-table-column prop="quantity" label="Current Qty" width="100" />
            <el-table-column prop="minStock" label="Min Stock" width="100" />
            <el-table-column prop="warehouseName" label="Warehouse" width="150" />
            <el-table-column prop="turnoverRate" label="Turnover Rate" width="120">
              <template #default="{ row }">
                {{ (row.turnoverRate * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="lastUpdateTime" label="Last Update" width="180">
              <template #default="{ row }">
                {{ row.lastUpdateTime ? new Date(row.lastUpdateTime).toLocaleString() : '-' }}
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Overstock" name="overstock">
          <el-table v-loading="loading" :data="overstockList" stripe>
            <el-table-column prop="productId" label="Product ID" width="120" />
            <el-table-column prop="productName" label="Product Name" min-width="150" show-overflow-tooltip />
            <el-table-column prop="quantity" label="Current Qty" width="100" />
            <el-table-column prop="maxStock" label="Max Stock" width="100" />
            <el-table-column prop="warehouseName" label="Warehouse" width="150" />
            <el-table-column prop="turnoverRate" label="Turnover Rate" width="120">
              <template #default="{ row }">
                {{ (row.turnoverRate * 100).toFixed(2) }}%
              </template>
            </el-table-column>
            <el-table-column prop="lastUpdateTime" label="Last Update" width="180">
              <template #default="{ row }">
                {{ row.lastUpdateTime ? new Date(row.lastUpdateTime).toLocaleString() : '-' }}
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="Category Sales" name="sales">
          <el-table v-loading="loading" :data="categorySalesList" stripe>
            <el-table-column prop="categoryId" label="Category ID" width="120" />
            <el-table-column prop="categoryName" label="Category Name" min-width="150" />
            <el-table-column prop="salesAmount" label="Sales Amount" width="150">
              <template #default="{ row }">
                ${{ row.salesAmount.toLocaleString() }}
              </template>
            </el-table-column>
            <el-table-column prop="salesQuantity" label="Sales Quantity" width="130" />
            <el-table-column prop="growthRate" label="Growth Rate" width="120">
              <template #default="{ row }">
                <span :class="row.growthRate >= 0 ? 'positive' : 'negative'">
                  {{ row.growthRate >= 0 ? '+' : '' }}{{ row.growthRate.toFixed(2) }}%
                </span>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.data-analysis-container {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.statistics-card {
  text-align: center;
}

.statistics-content {
  padding: 10px;
}

.statistics-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.statistics-value {
  font-size: 28px;
  font-weight: bold;
}

.statistics-value.warning {
  color: #e6a23c;
}

.statistics-value.danger {
  color: #f56c6c;
}

.statistics-label.warning {
  color: #e6a23c;
}

.statistics-label.danger {
  color: #f56c6c;
}

.query-card {
  margin-bottom: 20px;
}

.query-form :deep(.el-form-item) {
  margin-bottom: 0;
}

.table-card {
  margin-bottom: 20px;
}

.section-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 10px;
  color: #333;
}

.positive {
  color: #67c23a;
}

.negative {
  color: #f56c6c;
}
</style>