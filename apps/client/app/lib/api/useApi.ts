import { useCallback, useState } from 'react'
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

export function useApi<T>(
  apiCall: (...args: unknown[]) => Promise<T>,
  options: UseApiOptions<T> = {}
): UseApiReturn<T> {
  const [state, setState] = useState<UseApiState<T>>({
    loading: false,
    error: null,
    data: null,
  })

  const execute = useCallback(
    async (...args: unknown[]): Promise<T | null> => {
      setState((prev) => ({ ...prev, loading: true, error: null }))

      try {
        const result = await apiCall(...args)
        setState({ loading: false, error: null, data: result })

        if (options.onSuccess) {
          options.onSuccess(result)
        }

        return result
      } catch (error) {
        const apiError = isApiError(error)
          ? error
          : ApiError.fromError(error instanceof Error ? error : new Error(String(error)))

        setState({ loading: false, error: apiError, data: null })

        if (options.onError) {
          options.onError(apiError)
        }

        if (options.showErrorToast !== false) {
          console.error('API Error:', getErrorMessage(apiError))
        }

        return null
      }
    },
    [apiCall, options]
  )

  const reset = useCallback(() => {
    setState({ loading: false, error: null, data: null })
  }, [])

  return {
    ...state,
    execute,
    reset,
  }
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