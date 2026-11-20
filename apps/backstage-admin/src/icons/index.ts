import SvgIcon from '@/components/SvgIcon/index.vue' // SVG icon component
import type { App } from 'vue'

// Export the component registration function, used in main.js
export function setupSvgIcon(app: App) {
  app.component('svg-icon', SvgIcon)
}

// Auto-import all SVG icons with import.meta.glob
const modules = import.meta.glob('./svg/*.svg')

// Load all SVG modules
Object.keys(modules).forEach(key => {
  const module = modules[key]
  if (module) {
    module()
  }
})
