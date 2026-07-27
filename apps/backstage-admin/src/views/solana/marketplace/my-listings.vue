<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElTag } from 'element-plus'
import { Search, ArrowLeft, Delete } from '@element-plus/icons-vue'
import {
  getListingsAPI,
  delistListingAPI,
  type Listing,
} from '@/apis/solana'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()
const router = useRouter()

const sellerAddress = ref('')
const listings = ref<Listing[]>([])
const loading = ref(false)
const searching = ref(false)

async function handleSearch() {
  if (!sellerAddress.value) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  searching.value = true
  loading.value = true
  try {
    const res = await getListingsAPI(sellerAddress.value)
    listings.value = res.data || []
  } catch (e) {
    console.error('fetch my listings failed', e)
  } finally {
    loading.value = false
    searching.value = false
  }
}

async function handleDelist(listing: Listing) {
  loading.value = true
  try {
    await delistListingAPI(listing.listingAddress, listing.seller)
    listings.value = listings.value.filter(l => l.listingAddress !== listing.listingAddress)
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('delist failed', e)
  } finally {
    loading.value = false
  }
}

function statusTag(status: number) {
  if (status === 0) return { type: 'success' as const, label: t('solana.marketplace.active') }
  if (status === 1) return { type: 'info' as const, label: t('solana.marketplace.sold') }
  return { type: 'danger' as const, label: t('solana.marketplace.cancelled') }
}

function goBack() {
  router.push({ name: 'solanaMarketplace' })
}
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-title">
        <el-button text :icon="ArrowLeft" @click="goBack" />
        {{ t('solana.marketplace.myListingsTitle') }}
      </div>
    </div>

    <el-card shadow="never" style="margin-bottom: 16px">
      <el-form :inline="true" @submit.prevent="handleSearch">
        <el-form-item :label="t('solana.marketplace.sellerAddress')" required>
          <el-input
            v-model="sellerAddress"
            :placeholder="t('solana.marketplace.sellerPlaceholder')"
            style="width: 360px"
            clearable
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" :loading="searching" @click="handleSearch">
            {{ t('solana.marketplace.search') }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never">
      <el-table :data="listings" v-loading="loading" stripe style="width: 100%">
        <el-table-column prop="listingAddress" :label="t('solana.marketplace.listingAddress')" min-width="200" show-overflow-tooltip />
        <el-table-column prop="mint" :label="t('solana.marketplace.mint')" min-width="160" show-overflow-tooltip />
        <el-table-column prop="price" :label="t('solana.marketplace.price')" width="120" />
        <el-table-column prop="listedAmount" :label="t('solana.marketplace.listedAmount')" width="110" />
        <el-table-column :label="t('common.status')" width="100">
          <template #default="{ row }">
            <el-tag :type="statusTag(row.status).type" size="small">
              {{ statusTag(row.status).label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('common.createTime')" width="160">
          <template #default="{ row }">
            {{ formatDateTime(row.createdAt * 1000) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.actions')" width="120" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 0" link type="danger" size="small" :icon="Delete" @click="handleDelist(row)">
              {{ t('solana.marketplace.cancel') }}
            </el-button>
            <span v-else>-</span>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-if="!loading && listings.length === 0 && sellerAddress" :description="t('common.noData')" />
    </el-card>
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
