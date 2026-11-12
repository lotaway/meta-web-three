import { afterEach, describe, expect, it, vi } from 'vitest'
import { createApp, nextTick } from 'vue'
import { createPinia } from 'pinia'
import { createPersistedState } from 'pinia-plugin-persistedstate'
import { useUserStore } from '@/stores/user'
import { USER_PERSIST_KEY } from '@/utils/user-storage'

vi.mock('@/apis/admin', () => ({
  adminLoginAPI: vi.fn(async () => ({ data: { tokenHead: 'Bearer ', token: 'token-123' } })),
  getAdminInfoAPI: vi.fn(async () => ({
    data: { roles: ['ROLE_ADMIN'], menus: [], icon: 'avatar.png' },
  })),
  adminLogoutAPI: vi.fn(async () => ({ data: null })),
}))

function createStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial))
  return {
    data,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
  }
}

/** Install pinia on a real app so registered plugins are activated. */
function setup(initial: Record<string, string> = {}) {
  const storage = createStorage(initial)
  vi.stubGlobal('localStorage', storage)
  const pinia = createPinia()
  pinia.use(createPersistedState({ storage }))
  createApp({}).use(pinia)
  return storage
}

afterEach(() => vi.unstubAllGlobals())

describe('user store persistence', () => {
  it('writes only the allowlisted fields after a login', async () => {
    const storage = setup()
    const store = useUserStore()
    await store.userLogin({ username: 'admin', password: 'super-secret' })
    await nextTick()
    const raw = storage.getItem(USER_PERSIST_KEY) ?? ''
    expect(raw).not.toBe('')
    expect(raw).not.toContain('super-secret')
    expect(JSON.parse(raw).userInfo.token).toBe('Bearer token-123')
  })

  it('purges and never restores a password from legacy persisted state', () => {
    const legacy = JSON.stringify({
      userInfo: {
        username: 'admin',
        password: 'super-secret',
        token: 'Bearer legacy',
        roles: [],
        menus: [],
        avatar: '',
      },
    })
    const storage = setup({ [USER_PERSIST_KEY]: legacy })
    const store = useUserStore()
    expect(store.userInfo).not.toHaveProperty('password')
    expect(store.userInfo.token).toBe('Bearer legacy')
    expect(storage.getItem(USER_PERSIST_KEY)).not.toContain('super-secret')
  })
})
