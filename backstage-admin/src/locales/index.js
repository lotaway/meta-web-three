import { reactive } from 'vue';
import zhCN from './zh-CN';
import enUS from './en-US';
export const locales = {
    'zh-CN': zhCN,
    'en-US': enUS,
};
const currentLocale = reactive({
    value: localStorage.getItem('locale') || 'zh-CN'
});
export function getLocale() {
    return currentLocale.value;
}
export function setLocale(locale) {
    currentLocale.value = locale;
    localStorage.setItem('locale', locale);
}
export function t(key, locale) {
    const localeToUse = locale || currentLocale.value;
    const keys = key.split('.');
    let value = locales[localeToUse];
    for (const k of keys) {
        if (value && typeof value === 'object') {
            value = value[k];
        }
        else {
            return key;
        }
    }
    return typeof value === 'string' ? value : key;
}
const i18n = {
    global: {
        t,
        locale: currentLocale,
    }
};
export default i18n;
