<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElTag } from 'element-plus'
import { Plus, ShoppingCart, Delete, List } from '@element-plus/icons-vue'
import {
  createListingAPI,
  getListingsAPI,
  buyListingAPI,
  delistListingAPI,
  type Listing,
  type CreateListingRequest,
} from '@/apis/solana'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()
const router = useRouter()

const listings = ref<Listing[]>([])
const listLoading = ref(false)
const createDialogVisible = ref(false)
const buyDialogVisible = ref(false)
const dialogLoading = ref(false)

const formData = ref<CreateListingRequest>({
  sellerAddress: '',
  mintAddress: '',
  paymentMintAddress: 'So11111111111111111111111111111111111111112',
  price: 0,
  listedAmount: 1,
})

const buyForm = ref({
  listingAddress: '',
  buyerAddress: '',
})

async function fetchListings() {
  listLoading.value = true
  try {
    const res = await getListingsAPI()
    listings.value = res.data || []
  } catch (e) {
    console.error('fetch listings failed', e)
  } finally {
    listLoading.value = false
  }
}

onMounted(fetchListings)

function resetForm() {
  formData.value = {
    sellerAddress: '',
    mintAddress: '',
    paymentMintAddress: 'So11111111111111111111111111111111111111112',
    price: 0,
    listedAmount: 1,
  }
}

async function handleCreate() {
  if (!formData.value.sellerAddress || !formData.value.mintAddress || formData.value.price <= 0) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    const res = await createListingAPI(formData.value)
    listings.value.unshift(res.data)
    createDialogVisible.value = false
    resetForm()
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('create listing failed', e)
  } finally {
    dialogLoading.value = false
  }
}

function showBuyDialog(listing: Listing) {
  buyForm.value = { listingAddress: listing.listingAddress, buyerAddress: '' }
  buyDialogVisible.value = true
}

async function handleBuy() {
  if (!buyForm.value.buyerAddress) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    const res = await buyListingAPI(buyForm.value.listingAddress, buyForm.value.buyerAddress)
    if (res.data) {
      const idx = listings.value.findIndex(l => l.listingAddress === buyForm.value.listingAddress)
      if (idx >= 0) listings.value[idx] = res.data
    }
    buyDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('buy failed', e)
  } finally {
    dialogLoading.value = false
  }
}

async function handleDelist(listing: Listing) {
  dialogLoading.value = true
  try {
    await delistListingAPI(listing.listingAddress, listing.seller)
    listings.value = listings.value.filter(l => l.listingAddress !== listing.listingAddress)
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('delist failed', e)
  } finally {
    dialogLoading.value = false
  }
}

function statusTag(status: number) {
  if (status === 0) return { type: 'success' as const, label: t('solana.marketplace.active') }
  if (status === 1) return { type: 'info' as const, label: t('solana.marketplace.sold') }
  return { type: 'danger' as const, label: t('solana.marketplace.cancelled') }
}
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-title">
        <el-icon><ShoppingCart /></el-icon>
        {{ t('solana.marketplace.title') }}
      </div>
      <el-button type="primary" :icon="Plus" @click="createDialogVisible = true">
        {{ t('solana.marketplace.list') }}
      </el-button>
    </div>

    <el-card shadow="never" style="margin-bottom: 16px">
      <el-button :icon="List" @click="router.push({ name: 'solanaMyListings' })">
        {{ t('solana.marketplace.myListings') }}
      </el-button>
    </el-card>

    <el-card shadow="never">
      <el-table :data="listings" v-loading="listLoading" stripe style="width: 100%">
        <el-table-column prop="listingAddress" :label="t('solana.marketplace.listingAddress')" min-width="200" show-overflow-tooltip />
        <el-table-column prop="seller" :label="t('solana.marketplace.seller')" min-width="160" show-overflow-tooltip />
        <el-table-column prop="mint" :label="t('solana.marketplace.mint')" min-width="160" show-overflow-tooltip />
        <el-table-column prop="price" :label="t('solana.marketplace.price')" width="120" />
        <el-table-column prop="listedAmount" :label="t('solana.marketplace.listedAmount')" width="110" />
        <el-table-column prop="status" :label="t('common.status')" width="100">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status).type" size="small">
              {{ statusTag(row.status).label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" :label="t('common.createTime')" width="160">
          <template #default="{ row }">
            {{ formatDateTime(row.createdAt * 1000) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.actions')" width="180" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 0" link type="success" size="small" @click="showBuyDialog(row)">
              <el-icon><ShoppingCart /></el-icon> {{ t('solana.marketplace.buy') }}
            </el-button>
            <el-button v-if="row.status === 0" link type="danger" size="small" @click="handleDelist(row)">
              <el-icon><Delete /></el-icon> {{ t('solana.marketplace.cancel') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Create Listing Dialog -->
    <el-dialog v-model="createDialogVisible" :title="t('solana.marketplace.list')" width="500px">
      <el-form :model="formData" label-width="140px">
        <el-form-item :label="t('solana.marketplace.seller')" required>
          <el-input v-model="formData.sellerAddress" :placeholder="t('solana.marketplace.sellerPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('solana.marketplace.mint')" required>
          <el-input v-model="formData.mintAddress" :placeholder="t('solana.marketplace.mintPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('solana.marketplace.price')" required>
          <el-input-number v-model="formData.price" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item :label="t('solana.marketplace.listedAmount')" required>
          <el-input-number v-model="formData.listedAmount" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item :label="t('solana.marketplace.paymentMint')">
          <el-input v-model="formData.paymentMintAddress" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">
          {{ t('common.confirm') }}
        </el-button>
      </template>
    </el-dialog>

    <!-- Buy Dialog -->
    <el-dialog v-model="buyDialogVisible" :title="t('solana.marketplace.buy')" width="500px">
      <el-form :model="buyForm" label-width="120px">
        <el-form-item :label="t('solana.marketplace.listingAddress')">
          <el-input v-model="buyForm.listingAddress" disabled />
        </el-form-item>
        <el-form-item :label="t('solana.marketplace.buyer')" required>
          <el-input v-model="buyForm.buyerAddress" :placeholder="t('solana.marketplace.buyerPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="buyDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleBuy">
          {{ t('common.confirm') }}
        </el-button>
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
