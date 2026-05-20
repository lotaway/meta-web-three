import { useCallback, useState } from 'react'
import { cartApi, DEFAULT_USER_ID } from '@/api/generated'
import type { CartItemDTO } from '@/src/generated/api/models'

export interface UseCartResult {
  items: CartItemDTO[]
  loading: boolean
  error: string | null
  fetchCart: (userId?: number) => Promise<void>
  addItem: (item: CartItemDTO, userId?: number) => Promise<number>
  updateQuantity: (id: number, quantity: number, userId?: number) => Promise<number>
  removeItems: (ids: number[], userId?: number) => Promise<number>
  clearCart: (userId?: number) => Promise<number>
}

export function useCart(): UseCartResult {
  const [items, setItems] = useState<CartItemDTO[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchCart = useCallback(async (userId: number = DEFAULT_USER_ID) => {
    setLoading(true)
    setError(null)
    try {
      const response = await cartApi.list({ xUserId: userId })
      setItems(response.data ?? [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch cart')
    } finally {
      setLoading(false)
    }
  }, [])

  const addItem = useCallback(async (item: CartItemDTO, userId: number = DEFAULT_USER_ID) => {
    setLoading(true)
    setError(null)
    try {
      const response = await cartApi.add({
        xUserId: userId,
        cartItemDTO: item,
      })
      const cartItemId = response.data ?? 0
      if (cartItemId > 0) {
        await fetchCart(userId)
      }
      return cartItemId
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to add item')
      throw e
    } finally {
      setLoading(false)
    }
  }, [fetchCart])

  const updateQuantity = useCallback(
    async (id: number, quantity: number, userId: number = DEFAULT_USER_ID) => {
      setLoading(true)
      setError(null)
      try {
        const response = await cartApi.updateQuantity({
          xUserId: userId,
          id,
          quantity,
        })
        const result = response.data ?? 0
        if (result > 0) {
          await fetchCart(userId)
        }
        return result
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to update quantity')
        throw e
      } finally {
        setLoading(false)
      }
    },
    [fetchCart],
  )

  const removeItems = useCallback(
    async (ids: number[], userId: number = DEFAULT_USER_ID) => {
      setLoading(true)
      setError(null)
      try {
        const response = await cartApi._delete({
          xUserId: userId,
          ids,
        })
        const result = response.data ?? 0
        if (result > 0) {
          await fetchCart(userId)
        }
        return result
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to remove items')
        throw e
      } finally {
        setLoading(false)
      }
    },
    [fetchCart],
  )

  const clearCart = useCallback(async (userId: number = DEFAULT_USER_ID) => {
    setLoading(true)
    setError(null)
    try {
      const response = await cartApi.clear({ xUserId: userId })
      const result = response.data ?? 0
      if (result > 0) {
        setItems([])
      }
      return result
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to clear cart')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    items,
    loading,
    error,
    fetchCart,
    addItem,
    updateQuantity,
    removeItems,
    clearCart,
  }
}