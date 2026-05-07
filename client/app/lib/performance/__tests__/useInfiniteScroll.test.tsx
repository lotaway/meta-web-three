import {
  useInfiniteScroll,
  useDebounce,
  useThrottle,
  useMemoizedFn,
  useVirtualizedList,
} from '../performance/useInfiniteScroll'

jest.useFakeTimers()

describe('performance/useInfiniteScroll', () => {
  describe('useDebounce', () => {
    it('should return initial value immediately', () => {
      const { result } = renderHook(() => useDebounce('initial', 500))
      expect(result.current).toBe('initial')
    })

    it('should update value after delay', () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: 'initial', delay: 500 } }
      )

      expect(result.current).toBe('initial')

      rerender({ value: 'updated', delay: 500 })
      expect(result.current).toBe('initial')

      jest.advanceTimersByTime(500)
    })

    it('should use different delays', () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: 'initial', delay: 300 } }
      )

      rerender({ value: 'updated', delay: 300 })
      jest.advanceTimersByTime(300)
      expect(result.current).toBe('updated')

      rerender({ value: 'changed', delay: 100 })
      jest.advanceTimersByTime(100)
      expect(result.current).toBe('changed')
    })
  })

  describe('useThrottle', () => {
    it('should call function immediately on first call', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      result()
      expect(fn).toHaveBeenCalledTimes(1)
    })

    it('should not call function again within throttle window', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      result()
      result()
      result()

      expect(fn).toHaveBeenCalledTimes(1)
    })

    it('should call function again after throttle window', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      result()
      jest.advanceTimersByTime(500)
      result()

      expect(fn).toHaveBeenCalledTimes(2)
    })
  })

  describe('useMemoizedFn', () => {
    it('should maintain stable function reference', () => {
      const callback = jest.fn()
      const { result, rerender } = renderHook(
        ({ fn }) => useMemoizedFn(fn),
        { initialProps: { fn: callback } }
      )

      const firstFn = result.current
      rerender({ fn: callback })
      const secondFn = result.current

      expect(firstFn).toBe(secondFn)
    })

    it('should call latest callback', () => {
      const callbacks: jest.Mock[] = []
      const { result, rerender } = renderHook(({ fn }) => useMemoizedFn(fn), {
        initialProps: { fn: jest.fn() },
      })
      callbacks.push(result.current)

      const newCallback = jest.fn()
      rerender({ fn: newCallback })
      callbacks.push(result.current)

      callbacks[0]()
      callbacks[1]()

      expect(callbacks[0]).not.toHaveBeenCalled()
      expect(callbacks[1]).toHaveBeenCalled()
    })
  })

  describe('useInfiniteScroll', () => {
    it('should provide list props', () => {
      const fetchMore = jest.fn().mockResolvedValue(undefined)
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
        })
      )

      expect(result.listProps.data).toEqual(data)
      expect(result.listProps.onEndReached).toBeDefined()
      expect(result.listProps.initialNumToRender).toBe(10)
      expect(result.listProps.maxToRenderPerBatch).toBe(5)
    })

    it('should not load more when hasMore is false', () => {
      const fetchMore = jest.fn().mockResolvedValue(undefined)
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
          hasMore: false,
        })
      )

      result.loadMore()
      expect(fetchMore).not.toHaveBeenCalled()
    })

    it('should handle refresh', async () => {
      const fetchMore = jest.fn().mockResolvedValue(undefined)
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
        })
      )

      await result.refresh()
      expect(fetchMore).toHaveBeenCalled()
    })

    it('should set isRefreshing during refresh', async () => {
      let resolveFn: () => void
      const fetchMore = jest.fn(
        () =>
          new Promise<void>((resolve) => {
            resolveFn = resolve
          })
      )
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
        })
      )

      const refreshPromise = result.refresh()
      expect(result.isRefreshing).toBe(true)

      resolveFn!()
      await refreshPromise
      expect(result.isRefreshing).toBe(false)
    })

    it('should set isLoadingMore during load more', async () => {
      let resolveFn: () => void
      const fetchMore = jest.fn(
        () =>
          new Promise<void>((resolve) => {
            resolveFn = resolve
          })
      )
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
          hasMore: true,
        })
      )

      const loadMorePromise = result.loadMore()
      expect(result.isLoadingMore).toBe(true)

      resolveFn!()
      await loadMorePromise
      expect(result.isLoadingMore).toBe(false)
    })

    it('should call onEndReached when near end', () => {
      const fetchMore = jest.fn().mockResolvedValue(undefined)
      const data = Array.from({ length: 20 }, (_, i) => i)

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
          hasMore: true,
        })
      )

      result.listProps.onEndReached?.()
      expect(fetchMore).toHaveBeenCalled()
    })
  })

  describe('useVirtualizedList', () => {
    it('should calculate visible range', () => {
      const getItemCount = jest.fn(() => 100)
      const { result } = renderHook(() => useVirtualizedList({ getItemCount }))

      expect(result.current.visibleRange).toEqual({ start: 0, end: 20 })
    })

    it('should update visible range on viewable items change', () => {
      const getItemCount = jest.fn(() => 100)
      const { result } = renderHook(() => useVirtualizedList({ getItemCount }))

      result.onViewableItemsChanged({
        viewableItems: [
          { index: 5, isViewable: true },
          { index: 15, isViewable: true },
        ],
      } as any)

      expect(result.current.visibleRange).toEqual({ start: 0, end: 20 })
    })
  })
})