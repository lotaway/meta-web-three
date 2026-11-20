import { onMounted, onBeforeMount, onBeforeUnmount, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'

const { body } = document
const WIDTH = 1024
const RATIO = 3

type AppStore = ReturnType<typeof useAppStore>
type MobileDetector = () => boolean

function closeMobileSidebar(appStore: AppStore): void {
  if (appStore.device !== 'mobile' || !appStore.sidebar.opened) return
  appStore.closeSideBar(false)
}

function registerRouteWatch(route: ReturnType<typeof useRoute>, appStore: AppStore): void {
  // Watch route changes
  watch(
    () => route,
    () => closeMobileSidebar(appStore),
  )
}

// Determine whether accessed from a mobile device
function detectMobile(): boolean {
  const rect = body.getBoundingClientRect()
  return rect.width - RATIO < WIDTH
}

function handleResize(appStore: AppStore, isMobile: MobileDetector): void {
  if (document.hidden) return
  const mobile = isMobile()
  appStore.toggleDevice(mobile ? 'mobile' : 'desktop')
  if (mobile) appStore.closeSideBar(true)
}

function registerLifecycleHooks(resizeHandler: () => void, isMobile: MobileDetector, appStore: AppStore): void {
  // Called before the component mounts
  onBeforeMount(() => window.addEventListener('resize', resizeHandler))
  // Called after the component mounts
  onMounted(() => {
    if (!isMobile()) return
    appStore.toggleDevice('mobile')
    appStore.closeSideBar(true)
  })
  // Called before the component unmounts
  onBeforeUnmount(() => window.removeEventListener('resize', resizeHandler))
}

export default function useResizeHandler() {
  const route = useRoute()
  const appStore = useAppStore()
  const isMobile = detectMobile
  const resizeHandler = () => handleResize(appStore, isMobile)

  registerRouteWatch(route, appStore)
  registerLifecycleHooks(resizeHandler, isMobile, appStore)

  return {
    isMobile,
    resizeHandler,
  }
}
