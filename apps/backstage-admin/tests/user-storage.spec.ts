import { describe, expect, it } from 'vitest'
import { USER_PERSIST_KEY, purgePersistedPassword, type UserStorage } from '@/utils/user-storage'

type MemoryStorage = UserStorage & { data: Map<string, string> }

function createStorage(initial?: Record<string, string>): MemoryStorage {
  const data = new Map(Object.entries(initial ?? {}))
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      data.set(key, value)
    },
    removeItem: (key) => {
      data.delete(key)
    },
  }
}

function session(withPassword: boolean): string {
  return JSON.stringify({
    userInfo: {
      username: 'admin',
      ...(withPassword ? { password: 'super-secret' } : {}),
      token: 'Bearer token-123',
      roles: ['ROLE_ADMIN'],
      menus: [],
      avatar: 'avatar.png',
    },
  })
}

describe('purgePersistedPassword', () => {
  it('does nothing when no state is persisted', () => {
    const storage = createStorage()
    expect(purgePersistedPassword(storage)).toBe(false)
    expect(storage.data.size).toBe(0)
  })

  it('removes a nested password while keeping the session', () => {
    const storage = createStorage({ [USER_PERSIST_KEY]: session(true) })
    expect(purgePersistedPassword(storage)).toBe(true)
    const stored = storage.getItem(USER_PERSIST_KEY) ?? ''
    expect(stored).not.toContain('super-secret')
    const state = JSON.parse(stored)
    expect(state.userInfo).not.toHaveProperty('password')
    expect(state.userInfo.token).toBe('Bearer token-123')
    expect(state.userInfo.username).toBe('admin')
    expect(state.userInfo.roles).toEqual(['ROLE_ADMIN'])
  })

  it('also removes a top-level password from legacy state', () => {
    const raw = JSON.stringify({ password: 'super-secret', userInfo: { token: 'Bearer token-123' } })
    const storage = createStorage({ [USER_PERSIST_KEY]: raw })
    expect(purgePersistedPassword(storage)).toBe(true)
    const stored = storage.getItem(USER_PERSIST_KEY) ?? ''
    expect(stored).not.toContain('super-secret')
    expect(JSON.parse(stored)).not.toHaveProperty('password')
  })

  it('leaves an already clean state byte-for-byte unchanged', () => {
    const raw = session(false)
    const storage = createStorage({ [USER_PERSIST_KEY]: raw })
    expect(purgePersistedPassword(storage)).toBe(false)
    expect(storage.getItem(USER_PERSIST_KEY)).toBe(raw)
  })

  it('discards unparsable state', () => {
    const storage = createStorage({ [USER_PERSIST_KEY]: '{not-json' })
    expect(purgePersistedPassword(storage)).toBe(true)
    expect(storage.getItem(USER_PERSIST_KEY)).toBeNull()
  })

  it('is idempotent across repeated runs', () => {
    const storage = createStorage({ [USER_PERSIST_KEY]: session(true) })
    expect(purgePersistedPassword(storage)).toBe(true)
    expect(purgePersistedPassword(storage)).toBe(false)
  })

  it('supports a custom storage key', () => {
    const storage = createStorage({ legacyUser: session(true) })
    expect(purgePersistedPassword(storage, 'legacyUser')).toBe(true)
    expect(storage.getItem('legacyUser')).not.toContain('super-secret')
    expect(storage.getItem(USER_PERSIST_KEY)).toBeNull()
  })
})
