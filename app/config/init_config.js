/**
 * APP内部通用配置
 * @describe 根据不同客户进行配置、可更新
 * */
const initConfig = {
    //  dev开发模式，pro产品模式
    "environment": 'dev',
    // "environment": "pro",
    "devHost": "http://192.168.44.229:3002",
    // "devHost": "http://192.168.42.105:3002",
    //  "host": "http://192.168.44.229:10010",   // 客户域名，所有接口统一调用
    // "host": "http://192.168.44.90:85",
    // "host": "http://192.168.44.168:88",
    "host": "http://www.shopbest.com.cn",
    // "updateTime": "2016-10-10 16:05:20",    //    app最后更新时间（无须修改）
};

export {initConfig};