import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { queryKeys } from './queryClient'
import { apiClient, ApiError } from '../api'

interface UseProductsParams {
  pageNum?: number
  pageSize?: number
  keyword?: string
  categoryId?: number
}

function useInvalidatingMutation<TVariables, TResult>(
  mutationFn: (variables: TVariables) => Promise<TResult>,
  queryKey: readonly unknown[]
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey })
    },
  })
}

async function fetchCart() {
  const response = await apiClient.get('/mall-trade-service/cart/list')
  return response
}

async function fetchProfile() {
  const response = await apiClient.get('/mall-user-service/user/info')
  return response
}

export function useProducts(params?: UseProductsParams) {
  return useQuery({
    queryKey: queryKeys.products.list(params),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/product/list', { params } as any)
      return response
    },
  })
}

export function useProductDetail(id: number) {
  return useQuery({
    queryKey: queryKeys.products.detail(id),
    queryFn: async () => {
      const response = await apiClient.get(`/mall-catalog-service/product/${id}`)
      return response
    },
    enabled: !!id,
  })
}

export function useProductSearch(keyword: string) {
  return useQuery({
    queryKey: queryKeys.products.search(keyword),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/product/search', { params: { keyword } } as any)
      return response
    },
    enabled: keyword.length > 0,
  })
}

export function useCategories(parentId?: number) {
  return useQuery({
    queryKey: queryKeys.categories.list(parentId),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/product-category/children', { params: { parentId } } as any)
      return response
    },
  })
}

export function useBrands(params?: Record<string, any>) {
  return useQuery({
    queryKey: queryKeys.brands.list(params),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/brand/list', params)
      return response
    },
  })
}

export function useOrders(params?: Record<string, any>) {
  return useQuery({
    queryKey: queryKeys.orders.list(params),
    queryFn: async () => {
      const response = await apiClient.get('/mall-trade-service/order/list', params)
      return response
    },
  })
}

export function useOrderDetail(id: number) {
  return useQuery({
    queryKey: queryKeys.orders.detail(id),
    queryFn: async () => {
      const response = await apiClient.get(`/mall-trade-service/order/${id}`)
      return response
    },
    enabled: !!id,
  })
}

function useCartQuery() {
  return useQuery({
    queryKey: queryKeys.cart.items,
    queryFn: fetchCart,
  })
}

function useCartMutations() {
  const addItem = useInvalidatingMutation(
    (data: any) => apiClient.post('/mall-trade-service/cart/add', data),
    queryKeys.cart.items
  )
  const removeItem = useInvalidatingMutation(
    (itemId: number) => apiClient.post('/mall-trade-service/cart/delete', { id: itemId }),
    queryKeys.cart.items
  )
  const updateQuantity = useInvalidatingMutation(
    ({ itemId, quantity }: { itemId: number; quantity: number }) =>
      apiClient.get('/mall-trade-service/cart/update/quantity', { params: { id: itemId, quantity } } as any),
    queryKeys.cart.items
  )
  return { addItem, removeItem, updateQuantity }
}

export function useCart() {
  const cartQuery = useCartQuery()
  const { addItem, removeItem, updateQuantity } = useCartMutations()

  return {
    cart: cartQuery.data,
    isLoading: cartQuery.isLoading,
    error: cartQuery.error,
    addItem: addItem.mutateAsync,
    removeItem: removeItem.mutateAsync,
    updateQuantity: updateQuantity.mutateAsync,
    refetch: cartQuery.refetch,
  }
}

export function useHomeContent() {
  return useQuery({
    queryKey: queryKeys.home.content,
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/home/content')
      return response
    },
    staleTime: 1000 * 60 * 5,
  })
}

function useProfileQuery() {
  return useQuery({
    queryKey: queryKeys.user.profile,
    queryFn: fetchProfile,
  })
}

function useUpdateProfileMutation() {
  return useInvalidatingMutation(
    (data: any) => apiClient.put('/mall-user-service/user/info', data),
    queryKeys.user.profile
  )
}

export function useUserProfile() {
  const profileQuery = useProfileQuery()
  const updateProfile = useUpdateProfileMutation()

  return {
    profile: profileQuery.data,
    isLoading: profileQuery.isLoading,
    error: profileQuery.error,
    updateProfile: updateProfile.mutateAsync,
    refetch: profileQuery.refetch,
  }
}