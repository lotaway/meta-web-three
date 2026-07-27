<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Plus, Search, Coin, Delete } from '@element-plus/icons-vue'
import {
  createSolanaTokenAPI,
  getSolanaTokenAPI,
  mintSolanaTokenAPI,
  burnSolanaTokenAPI,
  type SolanaToken,
  type CreateTokenRequest,
} from '@/apis/solana'

const { t } = useI18n()

const tokenList = ref<SolanaToken[]>([])
const listLoading = ref(false)
const createDialogVisible = ref(false)
const detailDialogVisible = ref(false)
const mintDialogVisible = ref(false)
const dialogLoading = ref(false)

const selectedToken = ref<SolanaToken | null>(null)
const searchMintAddress = ref('')

const formData = ref<CreateTokenRequest>({
  name: '',
  symbol: '',
  uri: '',
  tokenType: 'TOKEN',
  supply: 1000000,
  ownerAddress: '',
})

const tokenTypeOptions = [
  { label: 'Token (同质化代币)', value: 'TOKEN' },
  { label: 'NFT (非同质化代币)', value: 'NFT' },
  { label: 'SFT (半同质化代币)', value: 'SFT' },
]

const mintForm = ref({
  mintAddress: '',
  recipient: '',
  amount: 0,
})

const burnForm = ref({
  mintAddress: '',
  amount: 0,
})

function resetForm() {
  formData.value = {
    name: '',
    symbol: '',
    uri: '',
    tokenType: 'TOKEN',
    supply: 1000000,
    ownerAddress: '',
  }
}

async function handleCreate() {
  if (!formData.value.name || !formData.value.symbol || !formData.value.ownerAddress) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    const res = await createSolanaTokenAPI(formData.value)
    tokenList.value.unshift(res.data)
    createDialogVisible.value = false
    resetForm()
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('create token failed', e)
  } finally {
    dialogLoading.value = false
  }
}

async function handleSearch() {
  if (!searchMintAddress.value) {
    ElMessage.warning(t('common.pleaseEnterMintAddress'))
    return
  }
  listLoading.value = true
  try {
    const res = await getSolanaTokenAPI(searchMintAddress.value)
    if (res.data) {
      const exists = tokenList.value.find(t => t.mintAddress === res.data.mintAddress)
      if (!exists) {
        tokenList.value.unshift(res.data)
      }
      selectedToken.value = res.data
      detailDialogVisible.value = true
    }
  } catch (e) {
    console.error('query token failed', e)
  } finally {
    listLoading.value = false
  }
}

function showDetail(token: SolanaToken) {
  selectedToken.value = token
  detailDialogVisible.value = true
}

function showMint(token: SolanaToken) {
  selectedToken.value = token
  mintForm.value = { mintAddress: token.mintAddress, recipient: '', amount: 1000 }
  mintDialogVisible.value = true
}

async function handleMint() {
  if (!mintForm.value.recipient || mintForm.value.amount <= 0) {
    ElMessage.warning(t('common.fillRequiredFields'))
    return
  }
  dialogLoading.value = true
  try {
    await mintSolanaTokenAPI(mintForm.value.mintAddress, mintForm.value.recipient, mintForm.value.amount)
    mintDialogVisible.value = false
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('mint failed', e)
  } finally {
    dialogLoading.value = false
  }
}

async function handleBurn(token: SolanaToken) {
  burnForm.value.mintAddress = token.mintAddress
  burnForm.value.amount = 1
  dialogLoading.value = true
  try {
    await burnSolanaTokenAPI(burnForm.value.mintAddress, burnForm.value.amount)
    ElMessage.success(t('common.operationSuccess'))
  } catch (e) {
    console.error('burn failed', e)
  } finally {
    dialogLoading.value = false
  }
}
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-title">
        <el-icon><Coin /></el-icon>
        {{ t('solana.token.title') }}
      </div>
      <el-button type="primary" :icon="Plus" @click="createDialogVisible = true">
        {{ t('solana.token.create') }}
      </el-button>
    </div>

    <!-- Search -->
    <el-card class="search-card" shadow="never">
      <el-form :inline="true">
        <el-form-item :label="t('solana.token.mintAddress')">
          <el-input v-model="searchMintAddress" :placeholder="t('solana.token.mintAddressPlaceholder')" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" :loading="listLoading" @click="handleSearch">
            {{ t('common.search') }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Token List -->
    <el-card shadow="never">
      <el-table :data="tokenList" v-loading="listLoading" stripe style="width: 100%">
        <el-table-column prop="mintAddress" :label="t('solana.token.mintAddress')" min-width="180" show-overflow-tooltip />
        <el-table-column prop="name" :label="t('solana.token.name')" width="120" />
        <el-table-column prop="symbol" :label="t('solana.token.symbol')" width="80" />
        <el-table-column prop="tokenType" :label="t('solana.token.type')" width="80">
          <template #default="{ row }">
            <el-tag :type="row.tokenType === 'NFT' ? 'warning' : row.tokenType === 'SFT' ? 'info' : 'primary'" size="small">
              {{ row.tokenType }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="decimals" :label="t('solana.token.decimals')" width="80" />
        <el-table-column prop="supply" :label="t('solana.token.supply')" width="120" />
        <el-table-column prop="owner" :label="t('solana.token.owner')" min-width="180" show-overflow-tooltip />
        <el-table-column :label="t('common.actions')" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="showDetail(row)">
              {{ t('common.detail') }}
            </el-button>
            <el-button link type="success" size="small" @click="showMint(row)">
              {{ t('solana.token.mint') }}
            </el-button>
            <el-button link type="danger" size="small" @click="handleBurn(row)">
              {{ t('solana.token.burn') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Create Dialog -->
    <el-dialog v-model="createDialogVisible" :title="t('solana.token.create')" width="560px" :close-on-click-modal="false">
      <el-form :model="formData" label-width="120px">
        <el-form-item :label="t('solana.token.name')" required>
          <el-input v-model="formData.name" />
        </el-form-item>
        <el-form-item :label="t('solana.token.symbol')" required>
          <el-input v-model="formData.symbol" />
        </el-form-item>
        <el-form-item :label="t('solana.token.type')" required>
          <el-select v-model="formData.tokenType" style="width: 100%">
            <el-option v-for="opt in tokenTypeOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('solana.token.supply')" v-if="formData.tokenType !== 'NFT'">
          <el-input-number v-model="formData.supply" :min="1" :max="999999999" style="width: 100%" />
        </el-form-item>
        <el-form-item :label="t('solana.token.uri')">
          <el-input v-model="formData.uri" :placeholder="t('solana.token.uriPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('solana.token.owner')" required>
          <el-input v-model="formData.ownerAddress" :placeholder="t('solana.token.ownerPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">
          {{ t('common.confirm') }}
        </el-button>
      </template>
    </el-dialog>

    <!-- Detail Dialog -->
    <el-dialog v-model="detailDialogVisible" :title="t('common.detail')" width="500px">
      <el-descriptions :column="1" border v-if="selectedToken">
        <el-descriptions-item :label="t('solana.token.mintAddress')">{{ selectedToken.mintAddress }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.name')">{{ selectedToken.name }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.symbol')">{{ selectedToken.symbol }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.type')">{{ selectedToken.tokenType }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.decimals')">{{ selectedToken.decimals }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.supply')">{{ selectedToken.supply }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.owner')">{{ selectedToken.owner }}</el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.uri')">
          <el-link type="primary" :href="selectedToken.uri" target="_blank" v-if="selectedToken.uri">
            {{ selectedToken.uri }}
          </el-link>
          <span v-else>-</span>
        </el-descriptions-item>
        <el-descriptions-item :label="t('solana.token.txSignature')">{{ selectedToken.txSignature }}</el-descriptions-item>
      </el-descriptions>
    </el-dialog>

    <!-- Mint Dialog -->
    <el-dialog v-model="mintDialogVisible" :title="t('solana.token.mint')" width="500px">
      <el-form :model="mintForm" label-width="120px">
        <el-form-item :label="t('solana.token.mintAddress')">
          <el-input v-model="mintForm.mintAddress" disabled />
        </el-form-item>
        <el-form-item :label="t('solana.token.recipient')" required>
          <el-input v-model="mintForm.recipient" :placeholder="t('solana.token.recipientPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('solana.token.amount')" required>
          <el-input-number v-model="mintForm.amount" :min="1" style="width: 100%" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="mintDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleMint">
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
.search-card {
  margin-bottom: 16px;
}
</style>
