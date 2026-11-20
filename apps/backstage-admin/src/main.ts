import { createApp } from 'vue'
import App from './App.vue'
import { createPinia } from 'pinia'
import router from './router'
import '@/styles/index.scss' // 引入全局css样式
import 'normalize.css/normalize.css' // 重置css统一浏览器默认样式
import { setupSvgIcon } from './icons'
import 'virtual:svg-icons-register' // 引入SVG图标注册器（关键步骤）
import '@/router/guard' // 引入vue-router的前置和后置守卫函数
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
