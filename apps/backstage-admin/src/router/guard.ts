import router from '@/router/index'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import { useUserStore } from '@/stores/user'
import usePermissionStore from '@/stores/permission'
import { ElMessage } from 'element-plus'

// 无需登陆的白名单路径
const whiteList = ['/login', '/register']

const isBlockchainEnabled = import.meta.env.VITE_BLOCKCHAIN_ENABLED === 'true'
const blockchainPaths = ['/blockchain', '/solana']

function isBlockchainPath(path: string): boolean {
  return blockchainPaths.some(prefix => path.startsWith(prefix))
}
// 配置路由前置守卫函数（每次路由跳转都会执行）
router.beforeEach((to, from, next) => {
  NProgress.start()
  const userStore = useUserStore()
  const permissionStore = usePermissionStore()
  if (userStore.userInfo.token) {
    if (to.path === '/login') {
      // 登录状态下访问login直接跳转到首页
      next({ path: '/' })
      NProgress.done()
    } else {
      if (permissionStore.addRouters.length === 0) {
        // 登录状态下无动态路由时根据menus生成动态路由
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
      // 未登录状态下白名单路径放行
      next()
    } else {
      // 未登录状态下非白名单路径跳转到登录页
      next('/login')
      NProgress.done()
    }
  }
})

// 配置路由后置函数守卫函数
router.afterEach(() => {
  NProgress.done()
})
