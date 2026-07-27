import http from '@/utils/http'

export interface SolanaToken {
  mintAddress: string
  name: string
  symbol: string
  uri: string
  tokenType: 'TOKEN' | 'NFT' | 'SFT'
  decimals: number
  supply: string
  owner: string
  txSignature: string
}

export interface CreateTokenRequest {
  name: string
  symbol: string
  uri: string
  tokenType: 'TOKEN' | 'NFT' | 'SFT'
  supply?: number
  ownerAddress: string
}

export function createSolanaTokenAPI(data: CreateTokenRequest) {
  return http<SolanaToken>({ url: '/api/v1/solana/tokens', method: 'post', data })
}

export function getSolanaTokenAPI(mintAddress: string) {
  return http<SolanaToken>({ url: `/api/v1/solana/tokens/${mintAddress}`, method: 'get' })
}

export function mintSolanaTokenAPI(mintAddress: string, recipient: string, amount: number) {
  return http<SolanaToken>({
    url: `/api/v1/solana/tokens/${mintAddress}/mint`,
    method: 'post',
    data: { recipient, amount },
  })
}

export function burnSolanaTokenAPI(mintAddress: string, amount: number) {
  return http<string>({
    url: `/api/v1/solana/tokens/${mintAddress}/burn`,
    method: 'post',
    data: { amount },
  })
}

// Marketplace types
export interface Listing {
  listingAddress: string
  seller: string
  mint: string
  paymentMint: string
  price: number
  listedAmount: number
  status: 0 | 1 | 2
  createdAt: number
  txSignature: string
}

export interface CreateListingRequest {
  sellerAddress: string
  mintAddress: string
  paymentMintAddress?: string
  price: number
  listedAmount: number
}

export interface BuyRequest {
  listingAddress: string
  buyerAddress: string
}

// Marketplace APIs
export function createListingAPI(data: CreateListingRequest) {
  return http<Listing>({ url: '/api/v1/solana/marketplace/listings', method: 'post', data })
}

export function getListingsAPI(seller?: string) {
  const params = seller ? { seller } : undefined
  return http<Listing[]>({ url: '/api/v1/solana/marketplace/listings', method: 'get', params })
}

export function getListingAPI(listingAddress: string) {
  return http<Listing>({ url: `/api/v1/solana/marketplace/listings/${listingAddress}`, method: 'get' })
}

export function buyListingAPI(listingAddress: string, buyerAddress: string) {
  return http<Listing>({
    url: `/api/v1/solana/marketplace/listings/${listingAddress}/buy`,
    method: 'post',
    data: { buyerAddress },
  })
}

export function delistListingAPI(listingAddress: string, sellerAddress: string) {
  return http<string>({
    url: `/api/v1/solana/marketplace/listings/${listingAddress}/delist`,
    method: 'post',
    data: { sellerAddress },
  })
}

// Activity types
export interface Activity {
  activityAddress: string
  authority: string
  startTime: number
  endTime: number
  entryFee: number
  rewardPercentages: number[]
  totalPool: number
  participantCount: number
  txSignature: string
}

export interface CreateActivityRequest {
  authority: string
  startTime: number
  endTime: number
  entryFee: number
  rewardPercentages?: number[]
  paymentMint?: string
}

// Activity APIs
export function createActivityAPI(data: CreateActivityRequest) {
  return http<Activity>({ url: '/api/v1/solana/activities', method: 'post', data })
}

export function getActivitiesAPI() {
  return http<Activity[]>({ url: '/api/v1/solana/activities', method: 'get' })
}

export function getActivityAPI(activityAddress: string) {
  return http<Activity>({ url: `/api/v1/solana/activities/${activityAddress}`, method: 'get' })
}

export function participateActivityAPI(activityAddress: string, participant: string) {
  return http<Activity>({
    url: `/api/v1/solana/activities/${activityAddress}/participate`,
    method: 'post',
    data: { participant },
  })
}

export function claimRewardAPI(activityAddress: string, winner: string, rank: number) {
  return http<string>({
    url: `/api/v1/solana/activities/${activityAddress}/claim`,
    method: 'post',
    data: { winner, rank },
  })
}

// Commission types
export interface Commission {
  account: string
  upline: string
  level: number
  downlineCount: number
}

// Commission APIs
export function setUplineAPI(target: string, upline: string) {
  return http<Commission>({
    url: '/api/v1/solana/commission/upline',
    method: 'post',
    data: { target, upline },
  })
}

export function getCommissionAPI(account: string) {
  return http<Commission>({ url: `/api/v1/solana/commission/${account}`, method: 'get' })
}
