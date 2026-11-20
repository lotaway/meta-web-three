import router from '@/router/index'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import { useUserStore } from '@/stores/user'
import usePermissionStore from '@/stores/permission'
import { ElMessage } from 'element-plus'
import { BLOCKCHAIN_ENABLED } from '@/utils/env'

// Whitelist paths that require no login
const whiteList = ['/login', '/register']

const isBlockchainEnabled = BLOCKCHAIN_ENABLED
const blockchainPaths = ['/blockchain', '/solana']

function isBlockchainPath(path: string): boolean {
  return blockchainPaths.some(prefix => path.startsWith(prefix))
}
// Configure the route before-guard (runs on every navigation)
router.beforeEach((to, from, next) => {
  NProgress.start()
  const userStore = useUserStore()
  const permissionStore = usePermissionStore()
  if (userStore.userInfo.token) {
    if (to.path === '/login') {
      // Redirect to home when accessing login while logged in
      next({ path: '/' })
      NProgress.done()
    } else {
      if (permissionStore.addRouters.length === 0) {
        // Generate dynamic routes from menus when logged in with no dynamic routes yet
        permissionStore.generateRoutes({
          menus: userStore.userInfo.menus,
          username: userStore.userInfo.username,
        })
        permissionStore.addRouters.forEach(route => {
          router.addRoute(route)
        })
        next({ ...to, replace: true })
      } else {
        if (!isBlockchainEnabled && isBlockchainPath(to.path)) {
          ElMessage.warning('区块链功能未启用 (VITE_BLOCKCHAIN_ENABLED=false)')
          next({ path: '/' })
          NProgress.done()
          return
        }
        next()
      }
    }
  } else {
    if (whiteList.indexOf(to.path) !== -1) {
      // Allow whitelist paths when not logged in
      next()
    } else {
      // Redirect non-whitelist paths to login when not logged in
      next('/login')
      NProgress.done()
    }
  }
})

// Configure the route after-guard
router.afterEach(() => {
  NProgress.done()
})
