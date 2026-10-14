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
      const response = await apiClient.get('/mall-catalog-service/products', { params } as any)
      return response
    },
  })
}

export function useProductDetail(id: number) {
  return useQuery({
    queryKey: queryKeys.products.detail(id),
    queryFn: async () => {
      const response = await apiClient.get(`/mall-catalog-service/products/${id}`)
      return response
    },
    enabled: !!id,
  })
}

export function useProductSearch(keyword: string) {
  return useQuery({
    queryKey: queryKeys.products.search(keyword),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/products/search', { params: { keyword } } as any)
      return response
    },
    enabled: keyword.length > 0,
  })
}

export function useCategories(parentId?: number) {
  return useQuery({
    queryKey: queryKeys.categories.list(parentId),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/categories', { params: { parentId } } as any)
      return response
    },
  })
}

export function useBrands(params?: Record<string, any>) {
  return useQuery({
    queryKey: queryKeys.brands.list(params),
    queryFn: async () => {
      const response = await apiClient.get('/mall-catalog-service/brands', { params } as any)
      return response
    },
  })
}

export function useOrders(params?: Record<string, any>) {
  return useQuery({
    queryKey: queryKeys.orders.list(params),
    queryFn: async () => {
      const response = await apiClient.get('/mall-trade-service/orders', params)
      return response
    },
  })
}

export function useOrderDetail(id: number) {
  return useQuery({
    queryKey: queryKeys.orders.detail(id),
    queryFn: async () => {
      const response = await apiClient.get(`/mall-trade-service/orders/${id}`)
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
      const response = await apiClient.get('/mall-trade-service/cart/items')
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
    mutationFn: (itemId: number) => apiClient.delete(`/mall-trade-service/cart/items/${itemId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cart.items })
    },
  })

  const updateQuantityMutation = useMutation({
    mutationFn: ({ itemId, quantity }: { itemId: number; quantity: number }) =>
      apiClient.put(`/mall-trade-service/cart/items/${itemId}`, { quantity }),
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
    mutationFn: (data: any) => apiClient.put('/mall-user-service/user/profile', data),
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