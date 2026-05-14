import Cookies from 'js-cookie';
// 设置cookie
export function setCookie(key, value, expires) {
    return Cookies.set(key, value, { expires: expires });
}
// 获取cookie
export function getCookie(key) {
    return Cookies.get(key);
}
