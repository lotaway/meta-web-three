import { act, renderHook } from '@testing-library/react-native'
import {
  useInfiniteScroll,
  useDebounce,
  useThrottle,
  useMemoizedFn,
  useVirtualizedList,
} from '../../../../src/lib/performance/useInfiniteScroll'

jest.useFakeTimers()

const advance = (ms: number) => {
  // React state updates scheduled by timers are only flushed inside act().
  act(() => {
    jest.advanceTimersByTime(ms)
  })
}

const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

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

      advance(500)
      expect(result.current).toBe('updated')
    })

    it('should use different delays', () => {
      const { result, rerender } = renderHook(
        ({ value, delay }) => useDebounce(value, delay),
        { initialProps: { value: 'initial', delay: 300 } }
      )

      rerender({ value: 'updated', delay: 300 })
      advance(300)
      expect(result.current).toBe('updated')

      rerender({ value: 'changed', delay: 100 })
      advance(100)
      expect(result.current).toBe('changed')
    })
  })

  describe('useThrottle', () => {
    it('should call function immediately on first call', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      act(() => {
        result.current()
      })
      expect(fn).toHaveBeenCalledTimes(1)
    })

    it('should not call function again within throttle window', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      act(() => {
        result.current()
        result.current()
        result.current()
      })

      expect(fn).toHaveBeenCalledTimes(1)
    })

    it('should call function again after throttle window', () => {
      const fn = jest.fn()
      const { result } = renderHook(() => useThrottle(fn, 500))

      act(() => {
        result.current()
      })
      advance(500)
      act(() => {
        result.current()
      })

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
      const firstCallback = jest.fn()
      const latestCallback = jest.fn()
      const { result, rerender } = renderHook(({ fn }) => useMemoizedFn(fn), {
        initialProps: { fn: firstCallback },
      })

      const memoizedFn = result.current
      memoizedFn()
      expect(firstCallback).toHaveBeenCalledTimes(1)

      rerender({ fn: latestCallback })
      // The wrapper identity stays stable while dispatching to the newest fn.
      expect(result.current).toBe(memoizedFn)
      memoizedFn()

      expect(latestCallback).toHaveBeenCalledTimes(1)
      expect(firstCallback).toHaveBeenCalledTimes(1)
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

      expect(result.current.listProps.data).toEqual(data)
      expect(result.current.listProps.onEndReached).toBeDefined()
      expect(result.current.listProps.initialNumToRender).toBe(10)
      expect(result.current.listProps.maxToRenderPerBatch).toBe(5)
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

      result.current.loadMore()
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

      await act(async () => {
        await result.current.refresh()
      })
      expect(fetchMore).toHaveBeenCalled()
      expect(result.current.isRefreshing).toBe(false)
    })

    it('should set isRefreshing during refresh', async () => {
      const { promise, resolve } = deferred()
      const fetchMore = jest.fn(() => promise)
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
        })
      )

      let refreshPromise!: Promise<void>
      act(() => {
        refreshPromise = result.current.refresh()
      })
      expect(result.current.isRefreshing).toBe(true)

      await act(async () => {
        resolve()
        await refreshPromise
      })
      expect(result.current.isRefreshing).toBe(false)
    })

    it('should set isLoadingMore during load more', async () => {
      const { promise, resolve } = deferred()
      const fetchMore = jest.fn(() => promise)
      const data = [1, 2, 3]

      const { result } = renderHook(() =>
        useInfiniteScroll({
          data,
          fetchMore,
          hasMore: true,
        })
      )

      let loadMorePromise!: Promise<void>
      act(() => {
        loadMorePromise = result.current.loadMore()
      })
      expect(result.current.isLoadingMore).toBe(true)

      await act(async () => {
        resolve()
        await loadMorePromise
      })
      expect(result.current.isLoadingMore).toBe(false)
    })

    it('should reset loading state when fetchMore rejects', async () => {
      const fetchMore = jest.fn().mockRejectedValue(new Error('network'))
      const { result } = renderHook(() =>
        useInfiniteScroll({ data: [1, 2, 3], fetchMore })
      )

      await act(async () => {
        await expect(result.current.refresh()).rejects.toThrow('network')
      })
      expect(result.current.isRefreshing).toBe(false)
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

      result.current.listProps.onEndReached?.()
      expect(fetchMore).toHaveBeenCalled()
    })
  })

  describe('useVirtualizedList', () => {
    const getItemCount = () => 100
    const getItem = (index: number) => index

    it('should calculate visible range', () => {
      const { result } = renderHook(() =>
        useVirtualizedList({ getItemCount, getItem })
      )

      expect(result.current.visibleRange).toEqual({ start: 0, end: 20 })
    })

    it('should update visible range on viewable items change', () => {
      const { result } = renderHook(() =>
        useVirtualizedList({ getItemCount, getItem })
      )

      act(() => {
        result.current.onViewableItemsChanged({
          viewableItems: [
            { index: 5, isViewable: true },
            { index: 15, isViewable: true },
          ],
        } as any)
      })

      expect(result.current.visibleRange).toEqual({ start: 0, end: 20 })
    })
  })
})
