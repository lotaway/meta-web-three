import { reactive, computed } from 'vue'
import zhCN from './zh-CN'
import enUS from './en-US'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'

export type LocaleType = 'zh-CN' | 'en-US'

export const locales: Record<LocaleType, Record<string, any>> = {
  'zh-CN': zhCN,
  'en-US': enUS,
}

// Element Plus locale mapping
export const elementLocales: Record<LocaleType, any> = {
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

export function t(key: string, locale?: LocaleType): string {
  const localeToUse = locale || currentLocale.value
  const keys = key.split('.')
  let value: any = locales[localeToUse]
  
  for (const k of keys) {
    if (value && typeof value === 'object') {
      value = value[k]
    } else {
      return key
    }
  }
  
  return typeof value === 'string' ? value : key
}

const i18n = {
  global: {
    t,
    locale: currentLocale,
  }
}

export default i18n