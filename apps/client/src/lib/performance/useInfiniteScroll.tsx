import { useCallback, useRef, useState, useEffect } from 'react'
import { FlatListProps, ViewToken } from 'react-native'

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

async function runFetch(
  fetchMore: () => Promise<void>,
  isLoadingRef: { current: boolean },
  setLoading: (loading: boolean) => void
): Promise<void> {
  if (isLoadingRef.current) return
  isLoadingRef.current = true
  setLoading(true)
  try {
    await fetchMore()
  } finally {
    // A rejected fetchMore must not leave the list stuck in a loading state
    // or permanently block future refresh/loadMore calls.
    setLoading(false)
    isLoadingRef.current = false
  }
}

function useScrollRefresh(fetchMore: () => Promise<void>, isLoadingRef: { current: boolean }) {
  const [isRefreshing, setIsRefreshing] = useState(false)
  const refresh = useCallback(async () => {
    await runFetch(fetchMore, isLoadingRef, setIsRefreshing)
  }, [fetchMore])
  return { isRefreshing, refresh }
}

function useScrollLoadMore(
  fetchMore: () => Promise<void>,
  isLoadingRef: { current: boolean },
  hasMore: boolean
) {
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const loadMore = useCallback(async () => {
    if (!hasMore) return
    await runFetch(fetchMore, isLoadingRef, setIsLoadingMore)
  }, [fetchMore, hasMore])
  return { isLoadingMore, loadMore }
}

function useEndReached(loadMore: () => void, hasMore: boolean, isLoadingMore: boolean) {
  return useCallback(() => {
    if (hasMore && !isLoadingMore) loadMore()
  }, [hasMore, isLoadingMore, loadMore])
}

function createListProps<T>(
  data: T[],
  onEndReached: () => void,
  isLoadingMore: boolean,
  initialNumToRender: number,
  maxToRenderPerBatch: number,
  windowSize: number
): Partial<FlatListProps<T>> {
  return {
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
}

export function useInfiniteScroll<T>({
  data,
  fetchMore,
  hasMore = true,
  initialNumToRender = 10,
  maxToRenderPerBatch = 5,
  windowSize = 5,
}: UseInfiniteScrollOptions<T>): UseInfiniteScrollReturn<T> {
  const isLoadingRef = useRef(false)
  const { isRefreshing, refresh } = useScrollRefresh(fetchMore, isLoadingRef)
  const { isLoadingMore, loadMore } = useScrollLoadMore(fetchMore, isLoadingRef, hasMore)
  const onEndReached = useEndReached(loadMore, hasMore, isLoadingMore)
  const listProps = createListProps(data, onEndReached, isLoadingMore, initialNumToRender, maxToRenderPerBatch, windowSize)

  return { listProps, refresh, loadMore, isRefreshing, isLoadingMore }
}

function LoadingIndicator() {
  return null
}

interface UseVirtualizedListOptions {
  getItemCount: () => number
  getItem: (index: number) => any
}

function toVisibleRange(viewableItems: ViewToken[], getItemCount: () => number) {
  const start = viewableItems[0].index ?? 0
  const end = viewableItems[viewableItems.length - 1].index ?? 0
  return {
    start: Math.max(0, start - 5),
    end: Math.min(getItemCount(), end + 5),
  }
}

export function useVirtualizedList({ getItemCount, getItem }: UseVirtualizedListOptions) {
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 })

  const onViewableItemsChanged = useCallback(({ viewableItems }: { viewableItems: ViewToken[] }) => {
    if (viewableItems.length === 0) return
    setVisibleRange(toVisibleRange(viewableItems, getItemCount))
  }, [getItemCount])

  const renderItem = useCallback(
    ({ item, index }: { item: any; index: number }) => {
      if (index < visibleRange.start || index > visibleRange.end) return null
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

function createThrottledCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  lastRun: { current: number | null }
): T {
  return ((...args: Parameters<T>) => {
    const now = Date.now()
    if (lastRun.current === null || now - lastRun.current >= delay) {
      lastRun.current = now
      return callback(...args)
    }
    return undefined
  }) as T
}

export function useThrottle<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  // `null` means the throttled function has not run yet. Seeding the ref with
  // `Date.now()` suppresses the very first call for a whole `delay` window,
  // which contradicts leading-edge throttle semantics.
  const lastRun = useRef<number | null>(null)
  return useCallback(createThrottledCallback(callback, delay, lastRun), [callback, delay])
}

export function useMemoizedFn<T extends (...args: any[]) => any>(fn: T): T {
  const ref = useRef(fn)
  ref.current = fn

  return useCallback((...args: Parameters<T>) => ref.current(...args), []) as T
}
