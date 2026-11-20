import { defineStore } from 'pinia'
import { shallowRef } from 'vue'
import { asyncRouterMap, constantRouterMap } from '@/router/index'
import type { UmsMenu } from '@/types/menu'
import type { RouteRecordExt } from '@/types/router'
// Use the plain helper here: vue-i18n's `Composer.t` pulls the full message
// schema into its generic signature and makes this module hit TS2589.
import { t } from '@/locales'
import { BLOCKCHAIN_ENABLED } from '@/utils/env'

function hasPermission(menus: UmsMenu[], route: RouteRecordExt) {
  if (route.name) {
    const currMenu = getMenu(route.name as string, menus)
    if (currMenu != null) {
      if (currMenu.title != null && currMenu.title !== '') {
        route.meta!.title = t(`menu.${currMenu.title}`) || currMenu.title
      }
      if (currMenu.icon != null && currMenu.title !== '') {
        route.meta!.icon = currMenu.icon
      }
      if (currMenu.hidden != null) {
        route.hidden = currMenu.hidden !== 0
      }
      if (currMenu.sort != null) {
        route.sort = currMenu.sort
      }
      return true
    } else {
      route.sort = 0
      if (route.hidden !== undefined && route.hidden === true) {
        route.sort = -1
        return true
      } else {
        return false
      }
    }
  } else {
    return true
  }
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

  // Generate the accessible route table
  const generateRoutes = (data: { menus: UmsMenu[]; username: string }) => {
    const { menus, username } = data
    const accessedRouters = asyncRouterMap.filter(v => {
      // In test mode the admin account returns all menus directly
      if (testMode && username === 'admin') return true
      if (hasPermission(menus, v)) {
        if (v.children && v.children.length > 0) {
          v.children = v.children.filter(child => {
            if (hasPermission(menus, child)) {
              return child
            }
            return false
          })
          return v
        } else {
          return v
        }
      }
      return false
    })
    // Blockchain feature toggle
    if (!isBlockchainEnabled) {
      filterBlockchainRoutes(accessedRouters)
    }
    // Sort menus
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
