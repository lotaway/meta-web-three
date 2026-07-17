import http from '@/utils/http'

export interface Lead {
  id?: number
  leadNo: string
  name: string
  company: string
  title: string
  email: string
  phone: string
  mobile: string
  source: string
  status: string
  score: number
  industry: string
  city: string
  description: string
  assignedTo: string
  createdAt?: string
}

export interface Opportunity {
  id?: number
  opportunityNo: string
  title: string
  leadId: number
  customerId: number
  contactId: number
  pipelineId: number
  stage: string
  amount: number
  probability: number
  expectedCloseDate: string
  actualCloseDate: string
  competitor: string
  description: string
  assignedTo: string
  createdAt?: string
}

export interface SalesPipeline {
  id?: number
  name: string
  description: string
  stages: string[]
  isDefault: boolean
  sortOrder: number
}

export interface CustomerServiceTicket {
  id?: number
  ticketNo: string
  title: string
  customerId: number
  contactId: number
  orderId: number
  type: string
  priority: string
  status: string
  assignedTo: string
  description: string
  resolution: string
  createdAt?: string
}

export interface Campaign {
  id?: number
  name: string
  description: string
  type: string
  status: string
  startDate: string
  endDate: string
  budget: number
  actualCost: number
  expectedRevenue: number
  leadsGenerated: number
  convertedCustomers: number
}

export interface Contact {
  id?: number
  firstName: string
  lastName: string
  email: string
  phone: string
  mobile: string
  position: string
  department: string
  customerId: number
  isPrimary: boolean
}

export interface PipelineSummary {
  stages: string[]
  counts: number[]
}

// Lead APIs
export const getLeadById = (id: number) => {
  return http<Lead>({ url: `/api/crm/leads/${id}`, method: 'get' })
}

export const listLeads = (params: { status?: string; source?: string; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Lead[]; total: number }>({ url: '/api/crm/leads/list', method: 'get', params })
}

export const createLead = (data: Partial<Lead>) => {
  return http<Lead>({ url: '/api/crm/leads', method: 'post', data })
}

export const updateLead = (data: Partial<Lead>) => {
  return http<Lead>({ url: '/api/crm/leads', method: 'put', data })
}

export const deleteLead = (id: number) => {
  return http<void>({ url: `/api/crm/leads/${id}`, method: 'delete' })
}

export const convertLead = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/leads/${id}/convert`, method: 'post', data: {} })
}

export const disqualifyLead = (id: number, reason: string) => {
  return http<void>({ url: `/api/crm/leads/${id}/disqualify`, method: 'post', params: { reason } })
}

// Opportunity APIs
export const getOpportunityById = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/opportunities/${id}`, method: 'get' })
}

export const listOpportunities = (params: { stage?: string; pipelineId?: number; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Opportunity[]; total: number }>({ url: '/api/crm/opportunities/list', method: 'get', params })
}

export const createOpportunity = (data: Partial<Opportunity>) => {
  return http<Opportunity>({ url: '/api/crm/opportunities', method: 'post', data })
}

export const updateOpportunity = (data: Partial<Opportunity>) => {
  return http<Opportunity>({ url: '/api/crm/opportunities', method: 'put', data })
}

export const deleteOpportunity = (id: number) => {
  return http<void>({ url: `/api/crm/opportunities/${id}`, method: 'delete' })
}

export const advanceStage = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/opportunities/${id}/advance`, method: 'post' })
}

export const closeWon = (id: number, data: { actualCloseDate: string; description?: string }) => {
  return http<Opportunity>({ url: `/api/crm/opportunities/${id}/close-won`, method: 'post' })
}

export const closeLost = (id: number, data: { reason: string; description?: string }) => {
  return http<Opportunity>({ url: `/api/crm/opportunities/${id}/close-lost`, method: 'post', params: { reason: data.reason } })
}

export const getPipelineSummary = () => {
  return http<PipelineSummary>({ url: '/api/crm/opportunities/summary', method: 'get' })
}

// Ticket APIs
export const getTicketById = (id: number) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/tickets/${id}`, method: 'get' })
}

export const listTickets = (params: { status?: string; priority?: string; type?: string; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: CustomerServiceTicket[]; total: number }>({ url: '/api/crm/tickets/list', method: 'get', params })
}

export const createTicket = (data: Partial<CustomerServiceTicket>) => {
  return http<CustomerServiceTicket>({ url: '/api/crm/tickets', method: 'post', data })
}

export const updateTicket = (data: Partial<CustomerServiceTicket>) => {
  return http<CustomerServiceTicket>({ url: '/api/crm/tickets', method: 'put', data })
}

export const deleteTicket = (id: number) => {
  return http<void>({ url: `/api/crm/tickets/${id}`, method: 'delete' })
}

export const assignTicket = (id: number, assignedTo: string) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/tickets/${id}/assign`, method: 'put', params: { assignedTo } })
}

export const updateTicketStatus = (id: number, status: string, resolution?: string) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/tickets/${id}/status`, method: 'put', params: { status } })
}

// Campaign APIs
export const getCampaignById = (id: number) => {
  return http<Campaign>({ url: `/api/crm/campaigns/${id}`, method: 'get' })
}

export const listCampaigns = (params: { status?: string; type?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Campaign[]; total: number }>({ url: '/api/crm/campaigns/list', method: 'get', params })
}

export const createCampaign = (data: Partial<Campaign>) => {
  return http<Campaign>({ url: '/api/crm/campaigns', method: 'post', data })
}

export const updateCampaign = (data: Partial<Campaign>) => {
  return http<Campaign>({ url: '/api/crm/campaigns', method: 'put', data })
}

export const deleteCampaign = (id: number) => {
  return http<void>({ url: `/api/crm/campaigns/${id}`, method: 'delete' })
}

// Contact APIs
export const getContactById = (id: number) => {
  return http<Contact>({ url: `/api/crm/contacts/${id}`, method: 'get' })
}

export const listContacts = (params: { customerId?: number; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Contact[]; total: number }>({ url: '/api/crm/contacts/list', method: 'get', params })
}

export const createContact = (data: Partial<Contact>) => {
  return http<Contact>({ url: '/api/crm/contacts', method: 'post', data })
}

export const updateContact = (data: Partial<Contact>) => {
  return http<Contact>({ url: '/api/crm/contacts', method: 'put', data })
}

export const deleteContact = (id: number) => {
  return http<void>({ url: `/api/crm/contacts/${id}`, method: 'delete' })
}
