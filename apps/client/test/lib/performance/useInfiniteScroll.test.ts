import { useDebounce, useThrottle, useMemoizedFn, useInfiniteScroll } from '../../../app/lib/performance/useInfiniteScroll'

jest.useFakeTimers()

describe('useDebounce', () => {
  it('should return initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('initial', 500))
    expect(result.current).toBe('initial')
  })

  it('should return debounced value after delay', async () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 500),
      { initialProps: { value: 'initial' } }
    )

    expect(result.current).toBe('initial')

    rerender({ value: 'updated' })
    expect(result.current).toBe('initial')

    jest.advanceTimersByTime(500)
  })
})

describe('useThrottle', () => {
  it('should call function immediately on first call', () => {
    const fn = jest.fn()
    const throttledFn = useThrottle(fn, 500)

    throttledFn()
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('should not call function again within throttle window', () => {
    const fn = jest.fn()
    const throttledFn = useThrottle(fn, 500)

    throttledFn()
    throttledFn()
    throttledFn()

    expect(fn).toHaveBeenCalledTimes(1)
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
})

describe('useInfiniteScroll', () => {
  it('should provide list props', () => {
    const fetchMore = jest.fn()
    const { result } = renderHook(() =>
      useInfiniteScroll({
        data: [1, 2, 3],
        fetchMore,
      })
    )

    expect(result.listProps.data).toEqual([1, 2, 3])
    expect(result.listProps.onEndReached).toBeDefined()
  })

  it('should handle refresh', async () => {
    const fetchMore = jest.fn().mockResolvedValue(undefined)
    const { result } = renderHook(() =>
      useInfiniteScroll({
        data: [1, 2, 3],
        fetchMore,
      })
    )

    result.refresh()
    expect(result.isRefreshing).toBe(true)
  })
})