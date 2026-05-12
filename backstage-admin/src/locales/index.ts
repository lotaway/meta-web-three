import zhCN from './zh-CN'
import enUS from './en-US'

export type LocaleType = 'zh-CN' | 'en-US'

export const locales = {
  'zh-CN': zhCN,
  'en-US': enUS,
}

export function t(key: string, locale: LocaleType = 'zh-CN'): string {
  const keys = key.split('.')
  let value: any = locales[locale]
  
  for (const k of keys) {
    if (value && typeof value === 'object') {
      value = value[k]
    } else {
      return key
    }
  }
  
  return typeof value === 'string' ? value : key
}

export default locales