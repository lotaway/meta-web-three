<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus } from '@element-plus/icons-vue'
import {
  getGroupBuyActivityListAPI,
  getShareRewardConfigListAPI,
  getDistributionRelationListAPI,
  getCommunityListAPI,
  getGroupBuyOrderListAPI,
  createGroupBuyActivityAPI,
  createShareRewardConfigAPI,
  createCommunityAPI,
  type GroupBuyActivity,
  type ShareRewardConfig,
  type DistributionRelation,
  type Community,
  type GroupBuyOrder
} from '@/apis/social'

const { t } = useI18n()

const activeTab = ref('groupBuy')

const groupBuyLoading = ref(false)
const groupBuyList = ref<GroupBuyActivity[]>([])
const groupBuyTotal = ref(0)

const shareLoading = ref(false)
const shareList = ref<ShareRewardConfig[]>([])
const shareTotal = ref(0)

const distributionLoading = ref(false)
const distributionList = ref<DistributionRelation[]>([])
const distributionTotal = ref(0)

const communityLoading = ref(false)
const communityList = ref<Community[]>([])
const communityTotal = ref(0)

const orderLoading = ref(false)
const orderList = ref<GroupBuyOrder[]>([])
const orderTotal = ref(0)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const dialogType = ref('')

const groupBuyFormData = ref({
  activityName: '',
  productId: 0,
  groupPrice: 0,
  originalPrice: 0,
  requiredCount: 2,
  startTime: '',
  endTime: ''
})

const shareFormData = ref({
  name: '',
  rewardType: 1,
  fixedAmount: 0,
  percentage: 0,
  maxRewardCount: 0,
  maxRewardAmount: 0,
  validFrom: '',
  validTo: ''
})

const communityFormData = ref({
  name: '',
  description: '',
  ownerId: 0,
  avatarUrl: '',
  maxMembers: 100
})

const getGroupBuyList = async () => {
  groupBuyLoading.value = true
  try {
    const response = await getGroupBuyActivityListAPI({ pageNum: 1, pageSize: 10 })
    groupBuyLoading.value = false
    groupBuyList.value = (response as any).data || []
    groupBuyTotal.value = (response as any).total || 0
  } catch (error) {
    groupBuyLoading.value = false
    ElMessage.error('Failed to load group buy activities')
  }
}

const getShareList = async () => {
  shareLoading.value = true
  try {
    const response = await getShareRewardConfigListAPI({ pageNum: 1, pageSize: 10 })
    shareLoading.value = false
    shareList.value = (response as any).data || []
    shareTotal.value = (response as any).total || 0
  } catch (error) {
    shareLoading.value = false
    ElMessage.error('Failed to load share reward configs')
  }
}

const getDistributionList = async () => {
  distributionLoading.value = true
  try {
    const response = await getDistributionRelationListAPI({ pageNum: 1, pageSize: 10 })
    distributionLoading.value = false
    distributionList.value = (response as any).data || []
    distributionTotal.value = (response as any).total || 0
  } catch (error) {
    distributionLoading.value = false
    ElMessage.error('Failed to load distribution relations')
  }
}

const getCommunityList = async () => {
  communityLoading.value = true
  try {
    const response = await getCommunityListAPI({ pageNum: 1, pageSize: 10 })
    communityLoading.value = false
    communityList.value = (response as any).data || []
    communityTotal.value = (response as any).total || 0
  } catch (error) {
    communityLoading.value = false
    ElMessage.error('Failed to load communities')
  }
}

const getOrderList = async () => {
  orderLoading.value = true
  try {
    const response = await getGroupBuyOrderListAPI({ pageNum: 1, pageSize: 10 })
    orderLoading.value = false
    orderList.value = (response as any).data || []
    orderTotal.value = (response as any).total || 0
  } catch (error) {
    orderLoading.value = false
    ElMessage.error('Failed to load orders')
  }
}

onMounted(() => {
  getGroupBuyList()
})

const handleTabChange = (tab: any) => {
  activeTab.value = tab
  if (tab === 'groupBuy') getGroupBuyList()
  else if (tab === 'share') getShareList()
  else if (tab === 'distribution') getDistributionList()
  else if (tab === 'community') getCommunityList()
  else if (tab === 'order') getOrderList()
}

const handleAdd = (type: string) => {
  dialogType.value = type
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    if (dialogType.value === 'groupBuy') {
      await createGroupBuyActivityAPI(groupBuyFormData.value)
      ElMessage.success('Group buy activity created')
    } else if (dialogType.value === 'share') {
      await createShareRewardConfigAPI({
        name: shareFormData.value.name,
        rewardType: shareFormData.value.rewardType,
        fixedAmount: shareFormData.value.fixedAmount,
        percentage: shareFormData.value.percentage,
        maxRewardCount: shareFormData.value.maxRewardCount,
        maxRewardAmount: shareFormData.value.maxRewardAmount,
        validFrom: shareFormData.value.validFrom,
        validTo: shareFormData.value.validTo
      })
      ElMessage.success('Share reward config created')
    } else if (dialogType.value === 'community') {
      await createCommunityAPI(communityFormData.value)
      ElMessage.success('Community created')
    }
    dialogVisible.value = false
    handleTabChange(activeTab.value)
  } catch (error) {
    ElMessage.error('Failed to create')
  } finally {
    dialogLoading.value = false
  }
}

const getStatusType = (status: string): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    'PENDING': 'info',
    'ACTIVE': 'success',
    'ENDED': 'warning',
    'CANCELLED': 'danger',
    'SUCCESS': 'success',
    'FAILED': 'danger',
    'REFUNDED': 'warning'
  }
  return map[status]
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return time.replace('T', ' ').substring(0, 19)
}
</script>

<template>
  <div class="social-commerce-container">
    <el-tabs v-model="activeTab" @tab-change="handleTabChange">
      <el-tab-pane label="Group Buy" name="groupBuy">
        <el-card>
          <div class="toolbar">
            <el-button type="primary" :icon="Plus" @click="handleAdd('groupBuy')">Add Activity</el-button>
          </div>
          <el-table v-loading="groupBuyLoading" :data="groupBuyList" border stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="activityName" label="Activity Name" min-width="150" />
            <el-table-column prop="productId" label="Product ID" width="100" />
            <el-table-column prop="groupPrice" label="Group Price" width="100">
              <template #default="{ row }">
                ${{ (row.groupPrice / 100).toFixed(2) }}
              </template>
            </el-table-column>
            <el-table-column prop="requiredCount" label="Required" width="80" />
            <el-table-column prop="currentCount" label="Current" width="80" />
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="endTime" label="End Time" width="160">
              <template #default="{ row }">
                {{ formatTime(row.endTime) }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Share Rewards" name="share">
        <el-card>
          <div class="toolbar">
            <el-button type="primary" :icon="Plus" @click="handleAdd('share')">Add Config</el-button>
          </div>
          <el-table v-loading="shareLoading" :data="shareList" border stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="configName" label="Config Name" min-width="150" />
            <el-table-column prop="rewardType" label="Reward Type" width="100" />
            <el-table-column prop="fixedAmount" label="Fixed Amount" width="120">
              <template #default="{ row }">
                ${{ (row.fixedAmount || 0) }}
              </template>
            </el-table-column>
            <el-table-column prop="percentage" label="Percentage" width="100">
              <template #default="{ row }">
                {{ (row.percentage || 0) * 100 }}%
              </template>
            </el-table-column>
            <el-table-column prop="status" label="Status" width="80">
              <template #default="{ row }">
                <el-tag :type="row.status === 1 ? 'success' : 'info'">{{ row.status === 1 ? 'Active' : 'Inactive' }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Distribution" name="distribution">
        <el-card>
          <el-table v-loading="distributionLoading" :data="distributionList" border stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="referrerId" label="Referrer ID" width="100" />
            <el-table-column prop="level" label="Level" width="80" />
            <el-table-column prop="rootReferrerId" label="Root Referrer" width="120" />
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="bindTime" label="Bind Time" width="160">
              <template #default="{ row }">
                {{ formatTime(row.bindTime) }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Community" name="community">
        <el-card>
          <div class="toolbar">
            <el-button type="primary" :icon="Plus" @click="handleAdd('community')">Create Community</el-button>
          </div>
          <el-table v-loading="communityLoading" :data="communityList" border stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="communityName" label="Name" min-width="150" />
            <el-table-column prop="ownerId" label="Owner ID" width="100" />
            <el-table-column prop="memberCount" label="Members" width="80" />
            <el-table-column prop="maxMembers" label="Max" width="80" />
            <el-table-column prop="inviteCode" label="Invite Code" width="120" />
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Orders" name="order">
        <el-card>
          <el-table v-loading="orderLoading" :data="orderList" border stripe>
            <el-table-column prop="id" label="Order ID" width="100" />
            <el-table-column prop="activityName" label="Activity" min-width="120" />
            <el-table-column prop="userId" label="User ID" width="100" />
            <el-table-column prop="productName" label="Product" min-width="120" />
            <el-table-column prop="quantity" label="Qty" width="60" />
            <el-table-column prop="totalAmount" label="Amount" width="100">
              <template #default="{ row }">
                ${{ (row.totalAmount / 100).toFixed(2) }}
              </template>
            </el-table-column>
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="createdAt" label="Created" width="160">
              <template #default="{ row }">
                {{ formatTime(row.createdAt) }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="dialogVisible" :title="`Add ${dialogType}`" width="500px" :close-on-click-modal="false">
      <el-form v-loading="dialogLoading" label-width="120px">
        <template v-if="dialogType === 'groupBuy'">
          <el-form-item label="Activity Name">
            <el-input v-model="groupBuyFormData.activityName" />
          </el-form-item>
          <el-form-item label="Product ID">
            <el-input-number v-model="groupBuyFormData.productId" :min="1" style="width: 100%" />
          </el-form-item>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="Group Price">
                <el-input-number v-model="groupBuyFormData.groupPrice" :min="0" style="width: 100%" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="Original Price">
                <el-input-number v-model="groupBuyFormData.originalPrice" :min="0" style="width: 100%" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-form-item label="Required Count">
            <el-input-number v-model="groupBuyFormData.requiredCount" :min="2" style="width: 100%" />
          </el-form-item>
        </template>

        <template v-if="dialogType === 'share'">
          <el-form-item label="Config Name">
            <el-input v-model="shareFormData.name" />
          </el-form-item>
          <el-form-item label="Reward Type">
            <el-select v-model="shareFormData.rewardType" style="width: 100%">
              <el-option label="Fixed Amount" :value="1" />
              <el-option label="Percentage" :value="2" />
            </el-select>
          </el-form-item>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="Fixed Amount">
                <el-input-number v-model="shareFormData.fixedAmount" :min="0" style="width: 100%" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="Percentage">
                <el-input-number v-model="shareFormData.percentage" :min="0" :max="1" :step="0.01" style="width: 100%" />
              </el-form-item>
            </el-col>
          </el-row>
        </template>

        <template v-if="dialogType === 'community'">
          <el-form-item label="Community Name">
            <el-input v-model="communityFormData.name" />
          </el-form-item>
          <el-form-item label="Description">
            <el-input v-model="communityFormData.description" type="textarea" :rows="2" />
          </el-form-item>
          <el-form-item label="Owner ID">
            <el-input-number v-model="communityFormData.ownerId" :min="1" style="width: 100%" />
          </el-form-item>
          <el-form-item label="Max Members">
            <el-input-number v-model="communityFormData.maxMembers" :min="1" style="width: 100%" />
          </el-form-item>
        </template>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.social-commerce-container {
  padding: 20px;
}

.toolbar {
  margin-bottom: 15px;
}
</style>