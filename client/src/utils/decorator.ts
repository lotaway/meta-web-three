import host from "../config/host"
import initConfig from "../config/init"
import Logger from "../utils/logger"
import crypto from "../utils/crypto"

interface StoreValue extends Object {
    updateTime: number
    data: object
}

interface ConvertObject extends Object {
    [prop: string]: any
}

interface ConvertArray extends Array<ConvertArray | ConvertObject> {
}

interface AttrArray extends Array<string | AttrArray> {

}


/**
 * 转化网址（缺少域名时添加）
 * @param url (String) 需要转化的地址
 * @param _host (String) 目标域名
 * @return url (String) 转化后的地址
 */
function convertUrl(url: string, _host = host.mainServer): string {
    if (/^\//.test(url))
        url = _host + url
    return url
}

/**
 * 在数据集中转化网址
 * @param data {Object|Array} 数据集
 * @param attrs {Array|String} 要检查的属性名
 * @param _host {string=} 使用的域名
 * @return {Object|Array} 返回的数据集
 */
function convertDataUrl(data: ConvertArray | ConvertObject, attrs: AttrArray | string | unknown = ['src', 'url'], _host?: string): typeof data {
    if (!attrs)
        throw new Error("没有设置需要转换的字段")
    if (data instanceof Array && data.length) {
        data = data.map(item => convertDataUrl(item, attrs))
    } else if (typeof data === 'object') {
        (attrs as AttrArray).forEach(attr => {
            if (attr instanceof Array && attr.length) {
                const firstAttr = attr[0] as string
                (data as ConvertObject)[firstAttr] && ((data as ConvertObject)[firstAttr] = convertDataUrl((data as ConvertObject)[firstAttr], attr.slice(1)))
            } else if ((data as ConvertObject)[attr as string]) {
                (data as ConvertObject)[attr as string] = convertUrl((data as ConvertObject)[attr as string])
            }
        })
    }
    return data
}

//  接口签名
export function getApiSign(apiUrl: string, data: { [key: string | number]: any }): string {
    const signApiValue = apiUrl.match(/[\w\W]*\/([^\/]+)\.api/)
    if (signApiValue === null)
        return ""
    let targetData = ""
    for (const key in data) {
        targetData += `&${key}=${data[key]}`
    }
    return crypto.Md5.init(targetData.replace(/[=&]/g, "") + signApiValue[1])
}

/**
 * 对象转化为formdata
 * @param {Object} object
 */
export function obj2FormData(object: { [key: string | number]: any }) {
    const formData = new FormData()
    Object.keys(object).forEach(key => {
        const value = object[key]
        if (Array.isArray(value)) {
            value.forEach((subValue, i) =>
                formData.append(key + `[${i}]`, subValue)
            )
        } else {
            formData.append(key, object[key])
        }
    })
    return formData
}

function getData<DataType extends StoreValue["data"]>(storeName: string): Promise<DataType | any> {
    return new Promise((resolve, reject) => {
        let strData = localStorage.getItem(storeName)
            , storeValue
        if (strData) {
            storeValue = JSON.parse(strData) as StoreValue;
            storeValue ? resolve(storeValue.data as DataType) : reject();
        } else {
            reject();
        }
    });
    // return Promise.reject();
}

function saveData(storeName: string, data: StoreValue["data"]): Promise<any> {
    try {
        localStorage.setItem(storeName, JSON.stringify(data));
    } catch (e) {
        (new Logger()).output("存储数据出错:" + JSON.stringify(e));
        return Promise.reject(e);
    }
    return Promise.resolve();
}

export default class Decorator {

    //  参数默认值装饰器
    static setDefaultArgs<T extends any[]>(...defaultArgs: T): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...realArgs: any[]) {
                const largerArgs = defaultArgs.length > realArgs.length ? defaultArgs : realArgs
                return originMethod.call(this, ...largerArgs.map((item, index) => {
                    const def = defaultArgs[index]
                    let real = realArgs[index]
                    if (def?.constructor === Object && real?.constructor === Object)
                        Object.keys(def).forEach(key => {
                            real[key] = real[key] ?? def[key]
                        })
                    return real ?? def
                }));
            };
        }
    }

    /**
     * 适配装饰器
     * @param adapter {function} 适配器
     */
    static useAdapter<ReturnType extends any>(adapter: (result2: any) => ReturnType): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...args: any[]) {
                const result = originMethod.call(this, ...args)
                if (result instanceof Promise)
                    return result.then(adapter)
                else
                    return adapter(result)
            }
        }
    }

    static setUrlWithHost(_host: string = host.mainServer, attrs: AttrArray | string | unknown = undefined, adapter = convertDataUrl): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...args: any[]) {
                const result = originMethod.call(this, ...args)
                if (result instanceof Promise)
                    return result.then((args: any) => adapter(args, attrs, _host))
                else
                    return adapter(result, attrs, _host)
            }
        }
    }

    /**
     * 缓存装饰器
     * @param storeName {string} 存储名称（默认使用类名+方法名）
     * @param saveTime {number} 存储有效期，毫秒为单位
     */
    static useCache(storeName?: string, saveTime = initConfig.cacheValidityTime) {
        return function (target: any, propName: string, descriptor: PropertyDescriptor) {
            const method = descriptor.value
            descriptor.value = function (...params: any[]) {
                const realStoreName = storeName ?? (target.name || target.constructor.name) + "." + propName
                return getData(realStoreName).then(res => res?.time && res.time > +new Date() ? res.data : Promise.reject("数据已过期")).catch(err => method.call(this, ...params).then((data: any) => saveData(realStoreName, {
                    time: saveTime ? +new Date() + saveTime : "",
                    data: data
                }).then(e => data)))
            }
        }
    }

    static sign(apiUrl: string, adapter = getApiSign): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...[firstArg, otherArgs]: any[]) {
                firstArg.sign = adapter(apiUrl, firstArg)
                return originMethod.call(this, firstArg, ...otherArgs)
            }
        }
    }
}
