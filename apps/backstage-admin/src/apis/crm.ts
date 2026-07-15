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
  return http<Lead>({ url: `/api/crm/lead/${id}`, method: 'get' })
}

export const listLeads = (params: { status?: string; source?: string; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Lead[]; total: number }>({ url: '/api/crm/lead/list', method: 'get', params })
}

export const createLead = (data: Partial<Lead>) => {
  return http<Lead>({ url: '/api/crm/lead', method: 'post', data })
}

export const updateLead = (data: Partial<Lead>) => {
  return http<Lead>({ url: '/api/crm/lead', method: 'put', data })
}

export const deleteLead = (id: number) => {
  return http<void>({ url: `/api/crm/lead/${id}`, method: 'delete' })
}

export const convertLead = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/lead/${id}/convert`, method: 'post' })
}

export const disqualifyLead = (id: number, reason: string) => {
  return http<void>({ url: `/api/crm/lead/${id}/disqualify`, method: 'post', data: { reason } })
}

// Opportunity APIs
export const getOpportunityById = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/opportunity/${id}`, method: 'get' })
}

export const listOpportunities = (params: { stage?: string; pipelineId?: number; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Opportunity[]; total: number }>({ url: '/api/crm/opportunity/list', method: 'get', params })
}

export const createOpportunity = (data: Partial<Opportunity>) => {
  return http<Opportunity>({ url: '/api/crm/opportunity', method: 'post', data })
}

export const updateOpportunity = (data: Partial<Opportunity>) => {
  return http<Opportunity>({ url: '/api/crm/opportunity', method: 'put', data })
}

export const deleteOpportunity = (id: number) => {
  return http<void>({ url: `/api/crm/opportunity/${id}`, method: 'delete' })
}

export const advanceStage = (id: number) => {
  return http<Opportunity>({ url: `/api/crm/opportunity/${id}/advance`, method: 'post' })
}

export const closeWon = (id: number, data: { actualCloseDate: string; description?: string }) => {
  return http<Opportunity>({ url: `/api/crm/opportunity/${id}/close-won`, method: 'post', data })
}

export const closeLost = (id: number, data: { reason: string; description?: string }) => {
  return http<Opportunity>({ url: `/api/crm/opportunity/${id}/close-lost`, method: 'post', data })
}

export const getPipelineSummary = () => {
  return http<PipelineSummary>({ url: '/api/crm/pipeline/summary', method: 'get' })
}

// Ticket APIs
export const getTicketById = (id: number) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/ticket/${id}`, method: 'get' })
}

export const listTickets = (params: { status?: string; priority?: string; type?: string; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: CustomerServiceTicket[]; total: number }>({ url: '/api/crm/ticket/list', method: 'get', params })
}

export const createTicket = (data: Partial<CustomerServiceTicket>) => {
  return http<CustomerServiceTicket>({ url: '/api/crm/ticket', method: 'post', data })
}

export const updateTicket = (data: Partial<CustomerServiceTicket>) => {
  return http<CustomerServiceTicket>({ url: '/api/crm/ticket', method: 'put', data })
}

export const deleteTicket = (id: number) => {
  return http<void>({ url: `/api/crm/ticket/${id}`, method: 'delete' })
}

export const assignTicket = (id: number, assignedTo: string) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/ticket/${id}/assign`, method: 'post', data: { assignedTo } })
}

export const updateTicketStatus = (id: number, status: string, resolution?: string) => {
  return http<CustomerServiceTicket>({ url: `/api/crm/ticket/${id}/status`, method: 'put', data: { status, resolution } })
}

// Campaign APIs
export const getCampaignById = (id: number) => {
  return http<Campaign>({ url: `/api/crm/campaign/${id}`, method: 'get' })
}

export const listCampaigns = (params: { status?: string; type?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Campaign[]; total: number }>({ url: '/api/crm/campaign/list', method: 'get', params })
}

export const createCampaign = (data: Partial<Campaign>) => {
  return http<Campaign>({ url: '/api/crm/campaign', method: 'post', data })
}

export const updateCampaign = (data: Partial<Campaign>) => {
  return http<Campaign>({ url: '/api/crm/campaign', method: 'put', data })
}

export const deleteCampaign = (id: number) => {
  return http<void>({ url: `/api/crm/campaign/${id}`, method: 'delete' })
}

// Contact APIs
export const getContactById = (id: number) => {
  return http<Contact>({ url: `/api/crm/contact/${id}`, method: 'get' })
}

export const listContacts = (params: { customerId?: number; keywords?: string; page?: number; pageSize?: number }) => {
  return http<{ records: Contact[]; total: number }>({ url: '/api/crm/contact/list', method: 'get', params })
}

export const createContact = (data: Partial<Contact>) => {
  return http<Contact>({ url: '/api/crm/contact', method: 'post', data })
}

export const updateContact = (data: Partial<Contact>) => {
  return http<Contact>({ url: '/api/crm/contact', method: 'put', data })
}

export const deleteContact = (id: number) => {
  return http<void>({ url: `/api/crm/contact/${id}`, method: 'delete' })
}
