import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

// The real router module imports Element Plus SCSS, which Vitest cannot load.
// Stub only the route maps the permission store needs.
vi.mock('@/router/index', () => ({
  default: {},
  constantRouterMap: [],
  asyncRouterMap: [
    { path: '/dashboard', name: 'Dashboard', meta: { title: 'dashboard.title' } },
    {
      path: '/blockchain',
      name: 'blockchain',
      redirect: '/blockchain/traceability',
      meta: { title: 'blockchain.title' },
      children: [{ path: 'traceability', name: 'blockchainTraceability' }],
    },
    {
      path: '/solana',
      name: 'solana',
      redirect: '/solana/token',
      meta: { title: 'solana.title' },
      children: [{ path: 'token', name: 'solanaToken' }],
    },
  ],
}))

vi.mock('@/locales', () => ({ t: (key: string) => key }))

import usePermissionStore from '@/stores/permission'
import { BLOCKCHAIN_ENABLED } from '@/utils/env'

describe('permission store blockchain gating', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('runs with the blockchain feature off by default', () => {
    expect(BLOCKCHAIN_ENABLED).toBe(false)
  })

  it('does not expose blockchain / solana menu entries when disabled', () => {
    const store = usePermissionStore()
    store.generateRoutes({ menus: [], username: 'admin' })

    const paths = store.addRouters.map(route => route.path)
    expect(paths).not.toContain('/blockchain')
    expect(paths).not.toContain('/solana')
  })

  it('keeps the normal (fiat) routes intact when disabled', () => {
    const store = usePermissionStore()
    store.generateRoutes({ menus: [], username: 'admin' })

    const paths = store.addRouters.map(route => route.path)
    expect(paths).toContain('/dashboard')
    expect(paths.length).toBeGreaterThan(0)
    for (const route of store.addRouters) {
      if (route.children) {
        expect(route.children.some(child => child.path.startsWith('/blockchain'))).toBe(false)
        expect(route.children.some(child => child.path.startsWith('/solana'))).toBe(false)
      }
    }
  })
})
