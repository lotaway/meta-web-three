import http from '@/utils/http'

export interface Developer {
  developerId: string; email: string; companyName: string; contactName: string;
  status: string; plan: string; createdAt: string
}
export function registerDeveloperAPI(data: Record<string, any>) { return http<Developer>({ url: '/developer/register', method: 'post', data }) }
export function getDeveloperAPI(id: string) { return http<Developer>({ url: `/developer/${id}`, method: 'get' }) }
export function listPendingDevelopersAPI() { return http<Developer[]>({ url: '/developer/admin/pending', method: 'get' }) }
export function listApprovedDevelopersAPI() { return http<Developer[]>({ url: '/developer/admin/approved', method: 'get' }) }
export function approveDeveloperAPI(id: string, reviewedBy: string, note?: string) { return http<Developer>({ url: `/developer/admin/${id}/approve`, method: 'post', data: { reviewedBy, note } }) }
export function rejectDeveloperAPI(id: string, reviewedBy: string, reason: string) { return http<Developer>({ url: `/developer/admin/${id}/reject`, method: 'post', data: { reviewedBy, reason } }) }
export function suspendDeveloperAPI(id: string, reason: string) { return http<Developer>({ url: `/developer/admin/${id}/suspend`, method: 'post', data: { reason } }) }
export function reactivateDeveloperAPI(id: string) { return http<Developer>({ url: `/developer/admin/${id}/reactivate`, method: 'post' }) }

export interface ApiKey {
  keyId: string; developerId: string; keyName: string; status: string;
  createdAt: string; expiresAt: string
}
export function createApiKeyAPI(developerId: string, data: Record<string, any>) { return http<ApiKey>({ url: `/developer/api-keys/${developerId}`, method: 'post', data }) }
export function getApiKeyAPI(keyId: string) { return http<ApiKey>({ url: `/developer/api-keys/${keyId}`, method: 'get' }) }
export function listDeveloperApiKeysAPI(developerId: string) { return http<ApiKey[]>({ url: `/developer/api-keys/developer/${developerId}`, method: 'get' }) }
export function disableApiKeyAPI(keyId: string) { return http<ApiKey>({ url: `/developer/api-keys/${keyId}/disable`, method: 'post' }) }
export function enableApiKeyAPI(keyId: string) { return http<ApiKey>({ url: `/developer/api-keys/${keyId}/enable`, method: 'post' }) }
export function revokeApiKeyAPI(keyId: string) { return http<ApiKey>({ url: `/developer/api-keys/${keyId}/revoke`, method: 'post' }) }

export interface Subscription {
  subscriptionId: string; developerId: string; apiName: string; tier: string;
  status: string; createdAt: string
}
export function createSubscriptionAPI(developerId: string, data: Record<string, any>) { return http<Subscription>({ url: `/developer/subscriptions/${developerId}`, method: 'post', data }) }
export function getSubscriptionAPI(id: string) { return http<Subscription>({ url: `/developer/subscriptions/${id}`, method: 'get' }) }
export function listDeveloperSubscriptionsAPI(developerId: string) { return http<Subscription[]>({ url: `/developer/subscriptions/developer/${developerId}`, method: 'get' }) }
export function cancelSubscriptionAPI(id: string) { return http<Subscription>({ url: `/developer/subscriptions/${id}/cancel`, method: 'post' }) }
export function listPendingSubscriptionsAPI() { return http<Subscription[]>({ url: '/developer/subscriptions/admin/pending', method: 'get' }) }
export function listActiveSubscriptionsAPI() { return http<Subscription[]>({ url: '/developer/subscriptions/admin/active', method: 'get' }) }
export function approveSubscriptionAPI(id: string, reviewedBy: string) { return http<Subscription>({ url: `/developer/subscriptions/admin/${id}/approve`, method: 'post', data: { reviewedBy } }) }
export function rejectSubscriptionAPI(id: string, reviewedBy: string, reason: string) { return http<Subscription>({ url: `/developer/subscriptions/admin/${id}/reject`, method: 'post', data: { reviewedBy, reason } }) }

export interface OAuthApp {
  clientId: string; developerId: string; appName: string; redirectUris: string;
  status: string; createdAt: string
}
export function createOAuthAppAPI(developerId: string, data: Record<string, any>) { return http<OAuthApp>({ url: `/developer/oauth/${developerId}`, method: 'post', data }) }
export function getOAuthAppAPI(clientId: string) { return http<OAuthApp>({ url: `/developer/oauth/${clientId}`, method: 'get' }) }
export function listDeveloperOAuthAppsAPI(developerId: string) { return http<OAuthApp[]>({ url: `/developer/oauth/developer/${developerId}`, method: 'get' }) }
export function updateOAuthAppAPI(clientId: string, data: Record<string, any>) { return http<OAuthApp>({ url: `/developer/oauth/${clientId}`, method: 'put', data }) }
export function disableOAuthAppAPI(clientId: string) { return http<OAuthApp>({ url: `/developer/oauth/${clientId}/disable`, method: 'post' }) }
export function enableOAuthAppAPI(clientId: string) { return http<OAuthApp>({ url: `/developer/oauth/${clientId}/enable`, method: 'post' }) }
export function deleteOAuthAppAPI(clientId: string) { return http<void>({ url: `/developer/oauth/${clientId}`, method: 'delete' }) }

export interface UsageStats {
  apiEndpoint: string; callCount: number; avgResponseTime: number; date: string
}
export function getDeveloperUsageAPI(developerId: string, startTime: string, endTime: string) { return http<UsageStats[]>({ url: `/developer/usage/${developerId}`, method: 'get', params: { startTime, endTime } }) }
export function getUsageSummaryAPI(developerId: string, startTime: string, endTime: string) { return http<Record<string, any>>({ url: `/developer/usage/${developerId}/summary`, method: 'get', params: { startTime, endTime } }) }
export function getCurrentUsageAPI(developerId: string) { return http<Record<string, any>>({ url: `/developer/usage/${developerId}/current`, method: 'get' }) }

export function getBillingSummaryAPI(developerId: string) { return http<Record<string, any>>({ url: '/developer/billing/summary', method: 'get', params: { developerId } }) }
export function getDocsOpenAPIAPI(baseUrl?: string) { return http<Record<string, any>>({ url: '/developer/docs/openapi', method: 'get', params: { baseUrl } }) }
export function getSDKSamplesAPI(language?: string) { return http<Record<string, string>>({ url: '/developer/docs/sdk-samples', method: 'get', params: { language } }) }
