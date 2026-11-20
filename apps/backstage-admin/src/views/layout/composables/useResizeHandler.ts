import { onMounted, onBeforeMount, onBeforeUnmount, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'

const { body } = document
const WIDTH = 1024
const RATIO = 3

export default function useResizeHandler() {
  const route = useRoute()
  const appStore = useAppStore()

  // Watch route changes
  watch(
    () => route,
    () => {
      if (appStore.device === 'mobile' && appStore.sidebar.opened) {
        appStore.closeSideBar(false)
      }
    },
  )

  // Determine whether accessed from a mobile device
  const isMobile = () => {
    const rect = body.getBoundingClientRect()
    return rect.width - RATIO < WIDTH
  }

  const resizeHandler = () => {
    if (!document.hidden) {
      const mobile = isMobile()
      appStore.toggleDevice(mobile ? 'mobile' : 'desktop')

      if (mobile) {
        appStore.closeSideBar(true)
      }
    }
  }

  // Called before the component mounts
  onBeforeMount(() => {
    window.addEventListener('resize', resizeHandler)
  })

  // Called after the component mounts
  onMounted(() => {
    const mobile = isMobile()
    if (mobile) {
      appStore.toggleDevice('mobile')
      appStore.closeSideBar(true)
    }
  })

  // Called before the component unmounts
  onBeforeUnmount(() => {
    window.removeEventListener('resize', resizeHandler)
  })

  return {
    isMobile,
    resizeHandler,
  }
}
