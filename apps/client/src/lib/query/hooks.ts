import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { queryKeys } from './queryClient'
import { apiClient, ApiError } from '../api'

interface UseProductsParams {
  pageNum?: number
  pageSize?: number
  keyword?: string
  categoryId?: number
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

export function useCart() {
  const queryClient = useQueryClient()

  const cartQuery = useQuery({
    queryKey: queryKeys.cart.items,
    queryFn: async () => {
      const response = await apiClient.get('/mall-trade-service/cart/list')
      return response
    },
  })

  const addItemMutation = useMutation({
    mutationFn: (data: any) => apiClient.post('/mall-trade-service/cart/add', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cart.items })
    },
  })

  const removeItemMutation = useMutation({
    mutationFn: (itemId: number) => apiClient.post('/mall-trade-service/cart/delete', { id: itemId }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cart.items })
    },
  })

  const updateQuantityMutation = useMutation({
    mutationFn: ({ itemId, quantity }: { itemId: number; quantity: number }) =>
      apiClient.get('/mall-trade-service/cart/update/quantity', { params: { id: itemId, quantity } } as any),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cart.items })
    },
  })

  return {
    cart: cartQuery.data,
    isLoading: cartQuery.isLoading,
    error: cartQuery.error,
    addItem: addItemMutation.mutateAsync,
    removeItem: removeItemMutation.mutateAsync,
    updateQuantity: updateQuantityMutation.mutateAsync,
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

export function useUserProfile() {
  const queryClient = useQueryClient()

  const profileQuery = useQuery({
    queryKey: queryKeys.user.profile,
    queryFn: async () => {
      const response = await apiClient.get('/mall-user-service/user/info')
      return response
    },
  })

  const updateProfileMutation = useMutation({
    mutationFn: (data: any) => apiClient.put('/mall-user-service/user/info', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.user.profile })
    },
  })

  return {
    profile: profileQuery.data,
    isLoading: profileQuery.isLoading,
    error: profileQuery.error,
    updateProfile: updateProfileMutation.mutateAsync,
    refetch: profileQuery.refetch,
  }
}