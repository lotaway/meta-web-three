import { onMounted, onBeforeMount, onBeforeUnmount, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'

const RATIO = 3
const COMPACT_WIDTH = 640
const EXPANDED_WIDTH = 1024

type AppStore = ReturnType<typeof useAppStore>
type Device = 'compact' | 'medium' | 'expanded'

function measureWidth(): number {
  return document.body.getBoundingClientRect().width - RATIO
}

export function resolveDevice(width: number): Device {
  if (width < COMPACT_WIDTH) return 'compact'
  if (width < EXPANDED_WIDTH) return 'medium'
  return 'expanded'
}

function detectDevice(): Device {
  return resolveDevice(measureWidth())
}

function applyDevice(device: Device, appStore: AppStore): void {
  appStore.toggleDevice(device === 'compact' ? 'mobile' : 'desktop')
  if (device === 'expanded') {
    appStore.openSideBar(true)
    return
  }
  appStore.closeSideBar(true)
}

function watchRoute(route: ReturnType<typeof useRoute>, appStore: AppStore): void {
  watch(
    () => route,
    () => {
      if (appStore.device === 'mobile') appStore.closeSideBar(false)
    },
  )
}

export default function useResizeHandler() {
  const route = useRoute()
  const appStore = useAppStore()
  let lastDevice: Device | null = null

  const resizeHandler = (): void => {
    if (document.hidden) return
    const device = detectDevice()
    if (device === lastDevice) return
    lastDevice = device
    applyDevice(device, appStore)
  }

  onBeforeMount(() => window.addEventListener('resize', resizeHandler))
  onMounted(resizeHandler)
  onBeforeUnmount(() => window.removeEventListener('resize', resizeHandler))
  watchRoute(route, appStore)

  return { resizeHandler }
}
