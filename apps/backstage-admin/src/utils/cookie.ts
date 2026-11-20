import Cookies from 'js-cookie'

// Set cookie
export function setCookie(key: string, value: string, expires: number) {
  return Cookies.set(key, value, { expires: expires })
}

// Get cookie
export function getCookie(key: string) {
  return Cookies.get(key)
}
