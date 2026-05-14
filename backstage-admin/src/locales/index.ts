import { reactive } from 'vue'
import zhCN from './zh-CN'
import enUS from './en-US'

export type LocaleType = 'zh-CN' | 'en-US'

export const locales: Record<LocaleType, Record<string, any>> = {
  'zh-CN': zhCN,
  'en-US': enUS,
}

const currentLocale = reactive<{ value: LocaleType }>({
  value: (localStorage.getItem('locale') as LocaleType) || 'zh-CN'
})

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