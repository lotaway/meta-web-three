import { reactive, ref } from 'vue'
import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', () => {
  // Sidebar state
  const sidebar = reactive({
    opened: true,
    withoutAnimation: false,
  })
  // Device state
  const device = ref('desktop')

  // Toggle sidebar
  function toggleSideBar() {
    sidebar.opened = !sidebar.opened
  }

  // Close sidebar
  function closeSideBar(withoutAnimation: boolean) {
    sidebar.opened = false
    sidebar.withoutAnimation = withoutAnimation
  }

  // Toggle device: desktop or mobile
  function toggleDevice(deviceType: string) {
    device.value = deviceType
  }

  return {
    sidebar,
    device,
    toggleSideBar,
    closeSideBar,
    toggleDevice,
  }
})
