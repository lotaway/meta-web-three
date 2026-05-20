import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 1000 * 60 * 5,
      staleTime: 1000 * 60 * 2,
      retry: 2,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchInterval: false,
      networkMode: 'offlineFirst',
    },
    mutations: {
      retry: 1,
      networkMode: 'offlineFirst',
    },
  },
})

export const queryKeys = {
  products: {
    all: ['products'] as const,
    list: (params?: Record<string, any>) => ['products', 'list', params] as const,
    detail: (id: number) => ['products', 'detail', id] as const,
    search: (keyword: string) => ['products', 'search', keyword] as const,
  },
  categories: {
    all: ['categories'] as const,
    list: (parentId?: number) => ['categories', 'list', parentId] as const,
    tree: ['categories', 'tree'] as const,
  },
  brands: {
    all: ['brands'] as const,
    list: (params?: Record<string, any>) => ['brands', 'list', params] as const,
    detail: (id: number) => ['brands', 'detail', id] as const,
  },
  orders: {
    all: ['orders'] as const,
    list: (params?: Record<string, any>) => ['orders', 'list', params] as const,
    detail: (id: number) => ['orders', 'detail', id] as const,
  },
  cart: {
    all: ['cart'] as const,
    items: ['cart', 'items'] as const,
  },
  user: {
    profile: ['user', 'profile'] as const,
    addresses: ['user', 'addresses'] as const,
  },
  home: {
    content: ['home', 'content'] as const,
    recommends: ['home', 'recommends'] as const,
  },
} as const