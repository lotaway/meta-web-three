import { render, RenderOptions, RenderResult } from '@testing-library/react-native'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React, { ReactElement, ReactNode } from 'react'
import { ApiClient, apiClient } from '../../../app/lib/api/client'

export * from '@testing-library/react-native'

const defaultQueryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      gcTime: 0,
    },
    mutations: {
      retry: false,
    },
  },
})

interface WrapperProps {
  children: ReactNode
}

function createWrapper() {
  const queryClient = defaultQueryClient
  return function Wrapper({ children }: WrapperProps) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    )
  }
}

export function renderWithProviders(
  ui: ReactElement,
  options?: RenderOptions
): RenderResult {
  return render(ui, {
    wrapper: createWrapper(),
    ...options,
  })
}

export function renderWithRouter(
  ui: ReactElement,
  options?: RenderOptions
): RenderResult {
  return render(ui, options)
}

export { act, waitFor }

export function createMockResponse<T>(
  data: T,
  code: string = '0000',
  message: string = '操作成功'
) {
  return {
    code,
    message,
    data,
    timestamp: Date.now(),
  }
}

export function createMockErrorResponse(
  message: string,
  code: string = '9999',
  data: any = null
) {
  return {
    code,
    message,
    data,
    timestamp: Date.now(),
  }
}

export function createPaginatedResponse<T>(
  data: T[],
  pageNum: number = 1,
  pageSize: number = 10,
  total: number = 100
) {
  return {
    code: '0000',
    message: '操作成功',
    data: {
      list: data,
      pageNum,
      pageSize,
      total,
      totalPages: Math.ceil(total / pageSize),
    },
    timestamp: Date.now(),
  }
}

export function mockApiClient() {
  const mock = {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    patch: jest.fn(),
  }

  jest.spyOn(apiClient, 'get').mockImplementation(mock.get)
  jest.spyOn(apiClient, 'post').mockImplementation(mock.post)
  jest.spyOn(apiClient, 'put').mockImplementation(mock.put)
  jest.spyOn(apiClient, 'delete').mockImplementation(mock.delete)
  jest.spyOn(apiClient, 'patch').mockImplementation(mock.patch)

  return mock
}

export function mockFetch(
  response: any,
  options: { ok?: boolean; status?: number; error?: Error } = {}
) {
  const { ok = true, status = 200, error } = options

  if (error) {
    return jest.fn().mockRejectedValue(error)
  }

  return jest.fn().mockResolvedValue({
    ok,
    status,
    json: () => Promise.resolve(response),
    headers: new Headers({ 'content-type': 'application/json' }),
  })
}

export function createMockStore<T>(initialState: T) {
  let state = initialState

  const getState = () => state

  const setState = (partial: Partial<T>) => {
    state = { ...state, ...partial }
  }

  const reset = () => {
    state = initialState
  }

  return { getState, setState, reset }
}

export async function waitForWithFakeTimers(
  callback: () => void | Promise<void>,
  timeout: number = 1000
) {
  await act(async () => {
    const result = callback()
    if (result instanceof Promise) {
      await result
    }
    jest.advanceTimersByTime(timeout)
  })
}

export function flushMicrotasks() {
  return act(async () => {
    jest.runAllTimers()
  })
}

export function createMockDate(mockDate: string = '2024-01-01T00:00:00.000Z') {
  const date = new Date(mockDate)
  jest.spyOn(global, 'Date').mockImplementation(() => date as any)
  jest.spyOn(Date, 'now').mockImplementation(() => date.getTime())
  return date
}

export function mockConsole() {
  const originalConsole = { ...console }
  beforeEach(() => {
    console.log = jest.fn()
    console.warn = jest.fn()
    console.error = jest.fn()
    console.info = jest.fn()
  })
  afterEach(() => {
    console.log = originalConsole.log
    console.warn = originalConsole.warn
    console.error = originalConsole.error
    console.info = originalConsole.info
  })
}

export function createQueryWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  })

  return {
    queryClient,
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    ),
  }
}

export { apiClient }
export type { WrapperProps }