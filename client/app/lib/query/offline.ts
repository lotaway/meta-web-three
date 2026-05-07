import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { QueryClient } from '@tanstack/react-query'
import React, { useState, useEffect } from 'react'

const asyncStoragePersister = createAsyncStoragePersister({
  storage: AsyncStorage,
  maxAge: 1000 * 60 * 60 * 24,
  key: 'react-query-cache',
})

interface OfflinePersistProviderProps {
  children: React.ReactNode
  client: QueryClient
}

export function OfflinePersistProvider({ children, client }: OfflinePersistProviderProps) {
  return (
    <PersistQueryClientProvider
      client={client}
      persistOptions={{ persister: asyncStoragePersister }}
    >
      {children}
    </PersistQueryClientProvider>
  )
}

export function useOfflineManager() {
  const [isOnline, setIsOnline] = useState(true)
  const [pendingMutations, setPendingMutations] = useState(0)

  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    if (typeof window !== 'undefined') {
      setIsOnline(navigator.onLine)

      window.addEventListener('online', handleOnline)
      window.addEventListener('offline', handleOffline)

      return () => {
        window.removeEventListener('online', handleOnline)
        window.removeEventListener('offline', handleOffline)
      }
    }
  }, [])

  const syncPending = async () => {
    console.log('Syncing pending mutations...')
    setPendingMutations(0)
  }

  return {
    isOnline,
    pendingMutations,
    syncPending,
    retryPending: syncPending,
  }
}

export function useCachedQuery<T>(
  key: readonly any[],
  queryFn: () => Promise<T>,
  options?: {
    enabled?: boolean
    staleTime?: number
    gcTime?: number
  }
) {
  const { isOnline } = useOfflineManager()

  return { isOnline }
}