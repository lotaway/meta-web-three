<template>
  <div class="crm-dashboard">
    <h2>{{ t('crm.dashboard.title') }}</h2>
    <el-row :gutter="20" class="summary-cards">
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="summary-card">
            <div class="summary-value">{{ stats.totalLeads }}</div>
            <div class="summary-label">{{ t('crm.dashboard.totalLeads') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="summary-card">
            <div class="summary-value">{{ stats.activeOpportunities }}</div>
            <div class="summary-label">{{ t('crm.dashboard.activeOpportunities') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="summary-card">
            <div class="summary-value">{{ stats.openTickets }}</div>
            <div class="summary-label">{{ t('crm.dashboard.openTickets') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="summary-card">
            <div class="summary-value">{{ stats.activeCampaigns }}</div>
            <div class="summary-label">{{ t('crm.dashboard.activeCampaigns') }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="dashboard-section">
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>{{ t('crm.dashboard.pipelineSummary') }}</span>
          </template>
          <div v-if="pipelineSummary.stages.length" class="pipeline-chart">
            <div v-for="(stage, index) in pipelineSummary.stages" :key="index" class="pipeline-bar">
              <span class="pipeline-label">{{ stage }}</span>
              <el-progress :percentage="pipelineSummary.counts[index]" :stroke-width="20" />
              <span class="pipeline-count">{{ pipelineSummary.counts[index] }}</span>
            </div>
          </div>
          <el-empty v-else :description="t('message.noData')" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>{{ t('crm.dashboard.recentLeads') }}</span>
          </template>
          <el-table :data="recentLeads" stripe size="small" max-height="240">
            <el-table-column prop="leadNo" :label="t('crm.lead.leadNo')" width="120" />
            <el-table-column prop="name" :label="t('crm.lead.name')" />
            <el-table-column prop="company" :label="t('crm.lead.company')" />
            <el-table-column prop="createdAt" :label="t('crm.lead.createdAt')" width="100">
              <template #default="{ row }">{{ row.createdAt?.slice(0, 10) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="dashboard-section">
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>{{ t('crm.dashboard.recentOpportunities') }}</span>
          </template>
          <el-table :data="recentOpportunities" stripe size="small" max-height="240">
            <el-table-column prop="opportunityNo" :label="t('crm.opportunity.opportunityNo')" width="120" />
            <el-table-column prop="title" :label="t('crm.opportunity.title')" />
            <el-table-column prop="amount" :label="t('crm.opportunity.amount')" width="100">
              <template #default="{ row }">{{ row.amount?.toLocaleString() }}</template>
            </el-table-column>
            <el-table-column prop="createdAt" :label="t('crm.opportunity.createdAt')" width="100">
              <template #default="{ row }">{{ row.createdAt?.slice(0, 10) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>{{ t('crm.dashboard.recentTickets') }}</span>
          </template>
          <el-table :data="recentTickets" stripe size="small" max-height="240">
            <el-table-column prop="ticketNo" :label="t('crm.ticket.ticketNo')" width="120" />
            <el-table-column prop="title" :label="t('crm.ticket.title')" />
            <el-table-column prop="priority" :label="t('crm.ticket.priority')" width="80">
              <template #default="{ row }">
                <el-tag :type="priorityType(row.priority)" size="small">{{ t('crm.ticketPriority.' + row.priority) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="status" :label="t('crm.ticket.status')" width="80">
              <template #default="{ row }">
                <el-tag :type="ticketStatusType(row.status)" size="small">{{ t('crm.ticketStatus.' + row.status) }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listLeads, listOpportunities, listTickets, listCampaigns, getPipelineSummary } from '@/apis/crm'
import type { Lead, Opportunity, CustomerServiceTicket, PipelineSummary } from '@/apis/crm'

const stats = ref({
  totalLeads: 0,
  activeOpportunities: 0,
  openTickets: 0,
  activeCampaigns: 0,
})

const pipelineSummary = ref<PipelineSummary>({ stages: [], counts: [] })
const recentLeads = ref<Lead[]>([])
const recentOpportunities = ref<Opportunity[]>([])
const recentTickets = ref<CustomerServiceTicket[]>([])

const priorityType = (p: string) => {
  const map: Record<string, string> = { LOW: 'info', MEDIUM: '', HIGH: 'warning', URGENT: 'danger' }
  return map[p] || 'info'
}

const ticketStatusType = (s: string) => {
  const map: Record<string, string> = { OPEN: 'info', ASSIGNED: 'primary', IN_PROGRESS: 'warning', RESOLVED: 'success', CLOSED: '' }
  return map[s] || 'info'
}

const fetchDashboardData = async () => {
  try {
    const [leadsRes, oppsRes, ticketsRes, campaignsRes, pipelineRes] = await Promise.all([
      listLeads({ page: 1, pageSize: 5 }),
      listOpportunities({ page: 1, pageSize: 5 }),
      listTickets({ page: 1, pageSize: 5 }),
      listCampaigns({ page: 1, pageSize: 100 }),
      getPipelineSummary(),
    ])
    recentLeads.value = leadsRes.data.records
    recentOpportunities.value = oppsRes.data.records
    recentTickets.value = ticketsRes.data.records
    stats.value.totalLeads = leadsRes.data.total
    stats.value.activeOpportunities = oppsRes.data.total
    stats.value.openTickets = ticketsRes.data.total
    stats.value.activeCampaigns = campaignsRes.data.total
    pipelineSummary.value = pipelineRes.data
  } catch (e) {
    console.error('Failed to fetch dashboard data:', e)
    ElMessage.error('Failed to load CRM dashboard data')
  }
}

onMounted(fetchDashboardData)
