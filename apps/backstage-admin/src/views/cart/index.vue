<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh, ShoppingCart } from '@element-plus/icons-vue'
import {
  getCartListWithPromotionAPI,
  addCartItemAPI,
  updateCartQuantityAPI,
  deleteCartItemsAPI,
  clearCartAPI,
  type CartItem
} from '@/apis/cart'

const { t } = useI18n()

// Query params
const listQuery = ref({
  pageNum: 1,
  pageSize: 10
})

// Data
const list = ref<CartItem[]>([])
const listLoading = ref(true)
const total = ref(0)

// Selected rows
const selectedRows = ref<CartItem[]>([])

// Dialog
const dialogVisible = ref(false)
const dialogLoading = ref(false)

// Add/Edit form
const formData = ref({
  productId: 0,
  productSkuId: 0,
  quantity: 1
})

// Promotion type options
const promotionTypeOptions = [
  { label: 'Discount', value: 'discount' },
  { label: 'Coupon', value: 'coupon' },
  { label: 'Flash Sale', value: 'flash' }
]

// Status options
const statusOptions = [
  { label: 'Normal', value: 0 },
  { label: 'Deleted', value: 1 }
]

// Computed
const totalAmount = computed(() => {
  return list.value.reduce((sum, item) => {
    const price = item.price || 0
    const discount = item.discountAmount || 0
    return sum + (price - discount) * item.quantity
  }, 0)
})

const totalItems = computed(() => {
  return list.value.reduce((sum, item) => sum + item.quantity, 0)
})

// Methods
const handleQuery = () => {
  listLoading.value = true
  getCartListWithPromotionAPI()
    .then((res) => {
      list.value = res || []
      total.value = list.value.length
    })
    .catch(() => {
      list.value = []
      total.value = 0
    })
    .finally(() => {
      listLoading.value = false
    })
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  handleQuery()
}

const handleSelectionChange = (rows: CartItem[]) => {
  selectedRows.value = rows
}

const handleAdd = () => {
  formData.value = {
    productId: 0,
    productSkuId: 0,
    quantity: 1
  }
  dialogVisible.value = true
}

const handleSubmit = () => {
  if (!formData.value.productId) {
    ElMessage.warning('Please select a product')
    return
  }
  if (formData.value.quantity <= 0) {
    ElMessage.warning('Quantity must be greater than 0')
    return
  }
  
  dialogLoading.value = true
  addCartItemAPI(formData.value)
    .then((res) => {
      ElMessage.success('Added successfully')
      dialogVisible.value = false
      handleQuery()
    })
    .catch(() => {
      ElMessage.error('Failed to add')
    })
    .finally(() => {
      dialogLoading.value = false
    })
}

const handleUpdateQuantity = (row: CartItem, quantity: number) => {
  if (!row.id) return
  if (quantity <= 0 || quantity === undefined) {
    ElMessage.warning('Quantity must be greater than 0')
    return
  }
  
  updateCartQuantityAPI(row.id, quantity)
    .then(() => {
      ElMessage.success('Updated successfully')
      handleQuery()
    })
    .catch(() => {
      ElMessage.error('Failed to update')
    })
}

const handleDelete = (row?: CartItem) => {
  const ids = row ? [row.id!] : selectedRows.value.map(r => r.id!).filter(Boolean)
  
  if (ids.length === 0) {
    ElMessage.warning('Please select items to delete')
    return
  }
  
  ElMessageBox.confirm(
    `Are you sure to delete ${row ? 'this item' : 'selected items'}?`,
    'Warning',
    {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    }
  ).then(() => {
    deleteCartItemsAPI(ids)
      .then(() => {
        ElMessage.success('Deleted successfully')
        handleQuery()
      })
      .catch(() => {
        ElMessage.error('Failed to delete')
      })
  }).catch(() => {})
}

const handleClear = () => {
  ElMessageBox.confirm(
    'Are you sure to clear all cart items?',
    'Warning',
    {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    }
  ).then(() => {
    clearCartAPI()
      .then(() => {
        ElMessage.success('Cleared successfully')
        handleQuery()
      })
      .catch(() => {
        ElMessage.error('Failed to clear')
      })
  }).catch(() => {})
}

const handleRefresh = () => {
  handleQuery()
}

const formatPrice = (price?: number) => {
  if (!price) return '¥0.00'
  return `¥${price.toFixed(2)}`
}

const formatDate = (date?: string) => {
  if (!date) return '-'
  return new Date(date).toLocaleString()
}

const getPromotionTypeLabel = (type?: string) => {
  const option = promotionTypeOptions.find(o => o.value === type)
  return option ? option.label : '-'
}

// Lifecycle
onMounted(() => {
  handleQuery()
})
</script>

<template>
  <div class="cart-container">
    <!-- Header -->
    <div class="header">
      <h2>
        <ShoppingCart class="icon" />
        Cart Management
      </h2>
      <div class="header-actions">
        <el-button type="primary" :icon="Plus" @click="handleAdd">
          Add Item
        </el-button>
        <el-button type="danger" :icon="Delete" :disabled="selectedRows.length === 0" @click="handleDelete()">
          Delete Selected
        </el-button>
        <el-button type="warning" :icon="Refresh" @click="handleClear">
          Clear Cart
        </el-button>
        <el-button :icon="Refresh" @click="handleRefresh">
          Refresh
        </el-button>
      </div>
    </div>

    <!-- Statistics -->
    <div class="statistics">
      <el-card class="stat-card">
        <div class="stat-label">Total Items</div>
        <div class="stat-value">{{ totalItems }}</div>
      </el-card>
      <el-card class="stat-card">
        <div class="stat-label">Total Amount</div>
        <div class="stat-value">{{ formatPrice(totalAmount) }}</div>
      </el-card>
      <el-card class="stat-card">
        <div class="stat-label">Cart Items</div>
        <div class="stat-value">{{ total }}</div>
      </el-card>
    </div>

    <!-- Table -->
    <el-table
      v-loading="listLoading"
      :data="list"
      border
      stripe
      @selection-change="handleSelectionChange"
      class="cart-table"
    >
      <el-table-column type="selection" width="55" />
      <el-table-column label="ID" prop="id" width="80" />
      <el-table-column label="Product" min-width="200">
        <template #default="{ row }">
          <div class="product-info">
            <img v-if="row.productPic" :src="row.productPic" class="product-pic" />
            <div class="product-detail">
              <div class="product-name">{{ row.productName }}</div>
              <div class="product-subtitle">{{ row.productSubTitle }}</div>
            </div>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="SKU Code" prop="productSkuCode" width="120" />
      <el-table-column label="Price" width="100">
        <template #default="{ row }">
          {{ formatPrice(row.price) }}
        </template>
      </el-table-column>
      <el-table-column label="Quantity" width="120">
        <template #default="{ row }">
          <el-input-number
            v-model="row.quantity"
            :min="1"
            :max="999"
            size="small"
            @change="(val?: number) => handleUpdateQuantity(row, val || 1)"
          />
        </template>
      </el-table-column>
      <el-table-column label="Discount" width="100">
        <template #default="{ row }">
          <span class="discount">{{ formatPrice(row.discountAmount) }}</span>
        </template>
      </el-table-column>
      <el-table-column label="Subtotal" width="120">
        <template #default="{ row }">
          <strong>{{ formatPrice((row.price - (row.discountAmount || 0)) * row.quantity) }}</strong>
        </template>
      </el-table-column>
      <el-table-column label="Promotion" width="150">
        <template #default="{ row }">
          <el-tag v-if="row.promotionTag" type="warning" size="small">
            {{ row.promotionTag }}
          </el-tag>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="Promotion Type" width="120">
        <template #default="{ row }">
          {{ getPromotionTypeLabel(row.promotionType) }}
        </template>
      </el-table-column>
      <el-table-column label="Member" width="120">
        <template #default="{ row }">
          {{ row.memberNickname || row.memberId }}
        </template>
      </el-table-column>
      <el-table-column label="Create Time" width="160">
        <template #default="{ row }">
          {{ formatDate(row.createDate) }}
        </template>
      </el-table-column>
      <el-table-column label="Actions" width="120" fixed="right">
        <template #default="{ row }">
          <el-button type="danger" size="small" :icon="Delete" @click="handleDelete(row)">
            Delete
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- Add/Edit Dialog -->
    <el-dialog
      v-model="dialogVisible"
      title="Add Cart Item"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Product ID" required>
          <el-input v-model.number="formData.productId" type="number" placeholder="Enter product ID" />
        </el-form-item>
        <el-form-item label="SKU ID" required>
          <el-input v-model.number="formData.productSkuId" type="number" placeholder="Enter SKU ID" />
        </el-form-item>
        <el-form-item label="Quantity" required>
          <el-input-number v-model="formData.quantity" :min="1" :max="999" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.cart-container {
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header h2 {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0;
}

.header .icon {
  font-size: 24px;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.statistics {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.stat-card {
  flex: 1;
  text-align: center;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 10px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409eff;
}

.cart-table {
  width: 100%;
}

.product-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.product-pic {
  width: 50px;
  height: 50px;
  object-fit: cover;
  border-radius: 4px;
  border: 1px solid #eee;
}

.product-detail {
  flex: 1;
  min-width: 0;
}

.product-name {
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.product-subtitle {
  font-size: 12px;
  color: #999;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.discount {
  color: #f56c6c;
}
</style>