import { useUserStore } from '@/stores/user';
import axios from 'axios';
import { ElMessage, ElMessageBox } from 'element-plus';
import { HTTP_TIMEOUT, MESSAGE_DURATION } from '@/constants';
import { t } from '@/locales';
const SERVICE_PREFIX_MAP = {
    '/product': 'product-service',
    '/brand': 'product-service',
    '/productCategory': 'product-service',
    '/productAttribute': 'product-service',
    '/sku': 'product-service',
    '/subject': 'product-service',
    '/prefrenceArea': 'product-service',
    '/order': 'order-service',
    '/returnReason': 'order-service',
    '/returnApply': 'order-service',
    '/companyAddress': 'order-service',
    '/orderSetting': 'order-service',
    '/coupon': 'promotion-service',
    '/flash': 'promotion-service',
    '/home': 'promotion-service',
    '/admin': 'user-service',
    '/role': 'user-service',
    '/menu': 'user-service',
    '/resource': 'user-service',
    '/memberLevel': 'user-service',
    '/sso': 'user-service',
    '/member': 'user-service',
};
const http = axios.create({
    baseURL: import.meta.env.VITE_BASE_SERVER_URL,
    timeout: HTTP_TIMEOUT,
});
http.interceptors.request.use(config => {
    const url = config.url || '';
    const matchedPrefix = Object.keys(SERVICE_PREFIX_MAP)
        .sort((a, b) => b.length - a.length)
        .find(prefix => url.startsWith(prefix) || url.startsWith('/' + prefix));
    if (matchedPrefix) {
        const serviceId = SERVICE_PREFIX_MAP[matchedPrefix];
        config.url = `/${serviceId}${url.startsWith('/') ? '' : '/'}${url}`;
    }
    return config;
}, e => Promise.reject(e));
http.interceptors.request.use(config => {
    const userStore = useUserStore();
    const token = userStore.userInfo.token;
    if (token) {
        config.headers.Authorization = token;
    }
    return config;
}, e => Promise.reject(e));
http.interceptors.response.use(response => {
    const res = response.data;
    const code = String(res.code);
    if (code !== '200') {
        ElMessage({
            message: res.message,
            type: 'error',
            duration: MESSAGE_DURATION,
        });
        if (code === '401') {
            ElMessageBox.confirm(t('http.logoutMessage'), t('http.logoutTitle'), {
                confirmButtonText: t('http.reLogin'),
                cancelButtonText: t('http.stayCurrentPage'),
                type: 'warning',
            }).then(() => {
                const userStore = useUserStore();
                userStore.fedLogout();
                location.reload();
            });
        }
        return Promise.reject('error');
    }
    else {
        return response.data;
    }
}, error => {
    ElMessage({
        message: error.message,
        type: 'error',
        duration: MESSAGE_DURATION,
    });
    return Promise.reject(error);
});
export default http;
