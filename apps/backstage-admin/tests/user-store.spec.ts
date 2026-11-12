import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useUserStore } from '@/stores/user'

vi.mock('@/apis/admin', () => ({
  adminLoginAPI: vi.fn(async () => ({ data: { tokenHead: 'Bearer ', token: 'token-123' } })),
  getAdminInfoAPI: vi.fn(async () => ({
    data: { roles: ['ROLE_ADMIN'], menus: [], icon: 'avatar.png' },
  })),
  adminLogoutAPI: vi.fn(async () => ({ data: null })),
}))

describe('user store credential handling', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('does not retain the submitted password in the store state', async () => {
    const store = useUserStore()
    await store.userLogin({ username: 'admin', password: 'super-secret' })
    expect(store.userInfo).not.toHaveProperty('password')
    expect(store.userInfo.token).toBe('Bearer token-123')
    expect(store.userInfo.username).toBe('admin')
  })

  it('keeps the password out of the serializable state used for persistence', async () => {
    const store = useUserStore()
    await store.userLogin({ username: 'admin', password: 'super-secret' })
    expect(JSON.stringify(store.$state)).not.toContain('super-secret')
  })

  it('resets the token on logout without introducing a password field', async () => {
    const store = useUserStore()
    await store.userLogin({ username: 'admin', password: 'super-secret' })
    store.fedLogout()
    expect(store.userInfo.token).toBe('')
    expect(store.userInfo).not.toHaveProperty('password')
  })
})
