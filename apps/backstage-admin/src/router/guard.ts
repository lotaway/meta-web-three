import router from '@/router/index'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import { useUserStore } from '@/stores/user'
import usePermissionStore from '@/stores/permission'
import { ElMessage } from 'element-plus'
import { BLOCKCHAIN_ENABLED } from '@/utils/env'
import type { NavigationGuardNext, RouteLocationNormalized } from 'vue-router'

// Whitelist paths that require no login
const whiteList = ['/login', '/register']

const isBlockchainEnabled = BLOCKCHAIN_ENABLED
const blockchainPaths = ['/blockchain', '/solana']

function isBlockchainPath(path: string): boolean {
  return blockchainPaths.some(prefix => path.startsWith(prefix))
}

type UserStore = ReturnType<typeof useUserStore>
type PermissionStore = ReturnType<typeof usePermissionStore>

function redirectHome(next: NavigationGuardNext): void {
  next({ path: '/' })
  NProgress.done()
}

function registerDynamicRoutes(
  userStore: UserStore,
  permissionStore: PermissionStore,
  to: RouteLocationNormalized,
  next: NavigationGuardNext,
): void {
  permissionStore.generateRoutes({
    menus: userStore.userInfo.menus,
    username: userStore.userInfo.username,
  })
  permissionStore.addRouters.forEach(route => { router.addRoute(route) })
  next({ ...to, replace: true })
}

function blockDisabledBlockchainPath(next: NavigationGuardNext): void {
  ElMessage.warning('区块链功能未启用 (VITE_BLOCKCHAIN_ENABLED=false)')
  redirectHome(next)
}

function handleAuthenticatedNavigation(
  to: RouteLocationNormalized,
  next: NavigationGuardNext,
  userStore: UserStore,
  permissionStore: PermissionStore,
): void {
  if (to.path === '/login') {
    // Redirect to home when accessing login while logged in
    redirectHome(next)
    return
  }
  if (permissionStore.addRouters.length === 0) {
    // Generate dynamic routes from menus when logged in with no dynamic routes yet
    registerDynamicRoutes(userStore, permissionStore, to, next)
    return
  }
  if (!isBlockchainEnabled && isBlockchainPath(to.path)) {
    blockDisabledBlockchainPath(next)
    return
  }
  next()
}

function handleGuestNavigation(to: RouteLocationNormalized, next: NavigationGuardNext): void {
  if (whiteList.indexOf(to.path) !== -1) {
    // Allow whitelist paths when not logged in
    next()
    return
  }
  // Redirect non-whitelist paths to login when not logged in
  next('/login')
  NProgress.done()
}

// Configure the route before-guard (runs on every navigation)
router.beforeEach((to, from, next) => {
  NProgress.start()
  const userStore = useUserStore()
  const permissionStore = usePermissionStore()
  if (userStore.userInfo.token) {
    handleAuthenticatedNavigation(to, next, userStore, permissionStore)
    return
  }
  handleGuestNavigation(to, next)
})

// Configure the route after-guard
router.afterEach(() => {
  NProgress.done()
})
