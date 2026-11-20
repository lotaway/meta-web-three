import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { QueryClient } from '@tanstack/react-query'
import React, { useState, useEffect } from 'react'

const asyncStoragePersister = createAsyncStoragePersister({
  storage: AsyncStorage,
  key: 'react-query-cache',
} as any)

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

const registerNetworkListeners = (onStatusChange: (online: boolean) => void) => {
  const handleOnline = () => onStatusChange(true)
  const handleOffline = () => onStatusChange(false)
  if (typeof window === 'undefined') return

  onStatusChange(navigator.onLine)
  window.addEventListener('online', handleOnline)
  window.addEventListener('offline', handleOffline)
  return () => {
    window.removeEventListener('online', handleOnline)
    window.removeEventListener('offline', handleOffline)
  }
}

export function useOfflineManager() {
  const [isOnline, setIsOnline] = useState(true)
  const [pendingMutations, setPendingMutations] = useState(0)

  useEffect(() => registerNetworkListeners(setIsOnline), [])

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