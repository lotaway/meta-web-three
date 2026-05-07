import { useCallback, useRef, useState, useEffect, DependencyList } from 'react'
import { FlatList, FlatListProps, ViewToken } from 'react-native'

interface UseInfiniteScrollOptions<T> {
  data: T[]
  fetchMore: () => Promise<void>
  hasMore?: boolean
  initialNumToRender?: number
  maxToRenderPerBatch?: number
  windowSize?: number
}

interface UseInfiniteScrollReturn<T> {
  listProps: Partial<FlatListProps<T>>
  refresh: () => void
  loadMore: () => void
  isRefreshing: boolean
  isLoadingMore: boolean
}

export function useInfiniteScroll<T>({
  data,
  fetchMore,
  hasMore = true,
  initialNumToRender = 10,
  maxToRenderPerBatch = 5,
  windowSize = 5,
}: UseInfiniteScrollOptions<T>): UseInfiniteScrollReturn<T> {
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const isLoadingRef = useRef(false)

  const refresh = useCallback(async () => {
    if (isLoadingRef.current) return
    isLoadingRef.current = true
    setIsRefreshing(true)
    await fetchMore()
    setIsRefreshing(false)
    isLoadingRef.current = false
  }, [fetchMore])

  const loadMore = useCallback(async () => {
    if (isLoadingRef.current || !hasMore) return
    isLoadingRef.current = true
    setIsLoadingMore(true)
    await fetchMore()
    setIsLoadingMore(false)
    isLoadingRef.current = false
  }, [fetchMore, hasMore])

  const onEndReached = useCallback(() => {
    if (hasMore && !isLoadingMore) {
      loadMore()
    }
  }, [hasMore, isLoadingMore, loadMore])

  const listProps: Partial<FlatListProps<T>> = {
    data,
    onEndReached,
    onEndReachedThreshold: 0.3,
    initialNumToRender,
    maxToRenderPerBatch,
    windowSize,
    removeClippedSubviews: true,
    keyExtractor: (item, index) => `${(item as any).id ?? index}`,
    ListFooterComponent: isLoadingMore ? () => <LoadingIndicator /> : null,
  }

  return { listProps, refresh, loadMore, isRefreshing, isLoadingMore }
}

function LoadingIndicator() {
  return null
}

interface UseVirtualizedListOptions {
  getItemCount: () => number
  getItem: (index: number) => any
}

export function useVirtualizedList({ getItemCount, getItem }: UseVirtualizedListOptions) {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 })

  const onViewableItemsChanged = useCallback(({ viewableItems }: { viewableItems: ViewToken[] }) => {
    if (viewableItems.length === 0) return

    const start = viewableItems[0].index ?? 0
    const end = viewableItems[viewableItems.length - 1].index ?? 0

    setVisibleRange({
      start: Math.max(0, start - 5),
      end: Math.min(getItemCount(), end + 5),
    })
  }, [getItemCount])

  const renderItem = useCallback(
    ({ item, index }: { item: any; index: number }) => {
      if (index < visibleRange.start || index > visibleRange.end) {
        return null
      }
      return { item }
    },
    [visibleRange]
  )

  return { visibleRange, onViewableItemsChanged, renderItem }
}

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => clearTimeout(handler)
  }, [value, delay])

  return debouncedValue
}

export function useThrottle<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const lastRun = useRef(Date.now())

  return useCallback(
    ((...args) => {
      const now = Date.now()
      if (now - lastRun.current >= delay) {
        lastRun.current = now
        return callback(...args)
      }
    }) as T,
    [callback, delay]
  )
}

export function useMemoizedFn<T extends (...args: any[]) => any>(fn: T): T {
  const ref = useRef(fn)
  ref.current = fn

  return useCallback((...args: any[]) => ref.current(...args), []) as T
}