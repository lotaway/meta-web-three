import { useCallback, useState, type Dispatch, type SetStateAction } from 'react'
import { apiClient } from './client'
import { ApiError, isApiError, getErrorMessage } from "./errors"

interface UseApiOptions<T> {
  onSuccess?: (data: T) => void
  onError?: (error: ApiError) => void
  showErrorToast?: boolean
}

interface UseApiState<T> {
  loading: boolean
  error: ApiError | null
  data: T | null
}

interface UseApiReturn<T> {
  execute: (...args: unknown[]) => Promise<T | null>
  loading: boolean
  error: ApiError | null
  data: T | null
  reset: () => void
}

function createInitialApiState<T>(): UseApiState<T> {
  return { loading: false, error: null, data: null }
}

function toApiError(error: unknown): ApiError {
  if (isApiError(error)) {
    return error
  }
  return ApiError.fromError(error instanceof Error ? error : new Error(String(error)))
}

function applyApiSuccess<T>(
  setState: Dispatch<SetStateAction<UseApiState<T>>>,
  options: UseApiOptions<T>,
  result: T
): void {
  setState({ loading: false, error: null, data: result })
  if (options.onSuccess) {
    options.onSuccess(result)
  }
}

function applyApiFailure<T>(
  setState: Dispatch<SetStateAction<UseApiState<T>>>,
  options: UseApiOptions<T>,
  error: unknown
): void {
  const apiError = toApiError(error)
  setState({ loading: false, error: apiError, data: null })
  if (options.onError) {
    options.onError(apiError)
  }
  if (options.showErrorToast !== false) {
    console.error('API Error:', getErrorMessage(apiError))
  }
}

function useApiExecute<T>(
  apiCall: (...args: unknown[]) => Promise<T>,
  options: UseApiOptions<T>,
  setState: Dispatch<SetStateAction<UseApiState<T>>>
) {
  return useCallback(
    async (...args: unknown[]): Promise<T | null> => {
      setState((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const result = await apiCall(...args)
        applyApiSuccess<T>(setState, options, result)
        return result
      } catch (error) {
        applyApiFailure(setState, options, error)
        return null
      }
    },
    [apiCall, options]
  )
}

export function useApi<T>(
  apiCall: (...args: unknown[]) => Promise<T>,
  options: UseApiOptions<T> = {}
): UseApiReturn<T> {
  const [state, setState] = useState<UseApiState<T>>(() => createInitialApiState<T>())
  const execute = useApiExecute<T>(apiCall, options, setState)
  const reset = useCallback(() => setState(createInitialApiState<T>()), [])

  return { ...state, execute, reset }
}

export function createApiCaller<T extends (...args: unknown[]) => Promise<unknown>>(
  apiCall: T
) {
  return useCallback(
    async (...args: Parameters<T>): Promise<Awaited<ReturnType<T>> | null> => {
      try {
        const result = await apiCall(...args)
        return result as Awaited<ReturnType<T>>
      } catch (error) {
        console.error('API Error:', getErrorMessage(error))
        return null
      }
    },
    [apiCall]
  )
}

export { apiClient, ApiError, isApiError, getErrorMessage }
export type { UseApiOptions, UseApiState, UseApiReturn }
