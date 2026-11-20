import { createApp } from 'vue'
import App from './App.vue'
import { createPinia } from 'pinia'
import router from './router'
import '@/styles/index.scss' // Import global CSS styles
import 'normalize.css/normalize.css' // Reset CSS to normalize browser defaults
import { setupSvgIcon } from './icons'
import 'virtual:svg-icons-register' // Import the SVG icon registrar (required step)
import '@/router/guard' // Import vue-router before/after guards
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import i18n from '@/locales'

const app = createApp(App)
const pinia = createPinia()
// Enable the persistence plugin for pinia
pinia.use(piniaPluginPersistedstate)
// Register pinia as a plugin
app.use(pinia)
// Register vue-router as a plugin
app.use(router)
// vue-i18n 11 (Composition API mode) must be installed before mounting:
// views call useI18n() and templates still use the injected $t helper.
app.use(i18n)
// Register the SvgIcon component as a plugin
setupSvgIcon(app)
app.mount('#app')
