import { defineStore } from 'pinia'
import { shallowRef } from 'vue'
import { asyncRouterMap, constantRouterMap } from '@/router/index'
import type { UmsMenu } from '@/types/menu'
import type { RouteRecordExt } from '@/types/router'
// Use the plain helper here: vue-i18n's `Composer.t` pulls the full message
// schema into its generic signature and makes this module hit TS2589.
import { t } from '@/locales'
import { BLOCKCHAIN_ENABLED } from '@/utils/env'

function applyMenuToRoute(menu: UmsMenu, route: RouteRecordExt) {
  if (menu.title != null && menu.title !== '') {
    route.meta!.title = t(`menu.${menu.title}`) || menu.title
  }
  if (menu.icon != null && menu.title !== '') {
    route.meta!.icon = menu.icon
  }
  if (menu.hidden != null) route.hidden = menu.hidden !== 0
  if (menu.sort != null) route.sort = menu.sort
}

function hasPermission(menus: UmsMenu[], route: RouteRecordExt) {
  if (!route.name) return true

  const currMenu = getMenu(route.name as string, menus)
  if (currMenu) {
    applyMenuToRoute(currMenu, route)
    return true
  }

  route.sort = route.hidden === true ? -1 : 0
  return route.hidden === true
}

// Get menu by route name
function getMenu(name: string, menus: UmsMenu[]) {
  return menus.find(menu => name === menu.name) || null
}

const isBlockchainEnabled = BLOCKCHAIN_ENABLED

function isBlockchainRoute(path: string): boolean {
  return path.startsWith('/blockchain') || path.startsWith('/solana')
}

function filterBlockchainRoutes(routes: RouteRecordExt[]): RouteRecordExt[] {
  return routes.filter(route => {
    if (isBlockchainRoute(route.path)) return false
    if (route.children) {
      route.children = route.children.filter(child => !isBlockchainRoute(child.path))
    }
    return true
  })
}

// Sort menus
function sortRouters(accessedRouters: RouteRecordExt[]) {
  accessedRouters.forEach(router => {
    if (router.children && router.children.length > 0) {
      router.children.sort((a, b) => compare(a, b))
    }
  })
  accessedRouters.sort((a, b) => compare(a, b))
}

// Descending comparison function
function compare(a: RouteRecordExt, b: RouteRecordExt) {
  if (a.sort && b.sort) {
    return b.sort - a.sort
  } else {
    return 0
  }
}

export const usePermissionStore = defineStore('permission', () => {
  // All routes: static + dynamic
  const routers = shallowRef(constantRouterMap)
  // Dynamic routes accessible with permission
  const addRouters = shallowRef<RouteRecordExt[]>([])
  // Whether test mode is on (enabled during development, returns all dynamic routes)
  const testMode = true

  const filterRouteByMenu = (route: RouteRecordExt, menus: UmsMenu[], username: string) => {
    if (testMode && username === 'admin') return true
    if (!hasPermission(menus, route)) return false
    if (route.children && route.children.length > 0) {
      route.children = route.children.filter(child => hasPermission(menus, child))
    }
    return true
  }

  // Generate the accessible route table
  const generateRoutes = (data: { menus: UmsMenu[]; username: string }) => {
    const { menus, username } = data
    let accessedRouters = asyncRouterMap.filter(v => filterRouteByMenu(v, menus, username))
    // filterBlockchainRoutes returns a NEW array (it drops top-level /blockchain
    // and /solana routes); the old code discarded it, leaving dead menu entries
    // when the feature is off. Reassign so the sidebar has no crypto entries.
    if (!isBlockchainEnabled) accessedRouters = filterBlockchainRoutes(accessedRouters)
    sortRouters(accessedRouters)
    addRouters.value = accessedRouters
    routers.value = constantRouterMap.concat(accessedRouters)
  }

  return {
    routers,
    addRouters,
    generateRoutes,
  }
})

// Default export for backward compatibility
export default usePermissionStore
