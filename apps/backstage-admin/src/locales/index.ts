import { reactive, computed, watch } from 'vue'
import { createI18n } from 'vue-i18n'
import type { Language } from 'element-plus/es/locale'
import zhCN from './zh-CN'
import enUS from './en-US'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'

export type LocaleType = 'zh-CN' | 'en-US'

export const locales = {
  'zh-CN': zhCN,
  'en-US': enUS,
}

// Element Plus locale mapping
export const elementLocales: Record<LocaleType, Language> = {
  'zh-CN': zhCn,
  'en-US': en,
}

const currentLocale = reactive<{ value: LocaleType }>({
  value: (localStorage.getItem('locale') as LocaleType) || 'zh-CN'
})

// Computed property for element-plus locale
export const currentElementLocale = computed(() => elementLocales[currentLocale.value])

export function getLocale(): LocaleType {
  return currentLocale.value
}

export function setLocale(locale: LocaleType): void {
  currentLocale.value = locale
  localStorage.setItem('locale', locale)
}

// Direct helper kept for non-component modules (stores, http interceptors).
// Components should use `useI18n()` instead, which also handles interpolation.
export function t(key: string, locale?: LocaleType): string {
  let value: unknown = locales[locale || currentLocale.value]

  for (const k of key.split('.')) {
    if (value && typeof value === 'object') {
      value = (value as Record<string, unknown>)[k]
    } else {
      return key
    }
  }

  return typeof value === 'string' ? value : key
}

// vue-i18n v11 requires a real, installed Composition API instance. Without
// one, the `useI18n()` calls in the views throw at runtime. `globalInjection`
// keeps the `$t` template helper working until the last legacy template is
// migrated, and `escapeParameter` hardens named interpolation.
export const i18n = createI18n({
  legacy: false,
  globalInjection: true,
  locale: currentLocale.value,
  fallbackLocale: 'zh-CN',
  messages: locales,
  escapeParameter: true,
  missingWarn: false,
  fallbackWarn: false,
})

// Keep the composer locale in sync with the reactive state used by the
// Element Plus config provider and the direct `t()` helper.
watch(
  () => currentLocale.value,
  (locale) => {
    i18n.global.locale.value = locale
  }
)

export default i18n
