import host from "../../config/host"
import initConfig from "../../config/init"
import crypto from "../../commons/utils/crypto"

interface CacheStoreValue extends Object {
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
        data = data.map(item => convertDataUrl(item, attrs, _host))
    } else if (typeof data === 'object') {
        (attrs as AttrArray).forEach(attr => {
            if (attr instanceof Array && attr.length) {
                const firstAttr = attr[0] as string
                (data as ConvertObject)[firstAttr] && ((data as ConvertObject)[firstAttr] = convertDataUrl((data as ConvertObject)[firstAttr], attr.slice(1), _host))
            } else if ((data as ConvertObject)[attr as string]) {
                (data as ConvertObject)[attr as string] = convertUrl((data as ConvertObject)[attr as string], _host)
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

function getData<DataType extends CacheStoreValue["data"]>(storeName: string): Promise<DataType | any> {
    return new Promise((resolve, reject) => {
        let strData = localStorage.getItem(storeName)
            , storeValue
        if (strData) {
            storeValue = JSON.parse(strData) as CacheStoreValue
            storeValue ? resolve(storeValue.data as DataType) : reject()
        } else {
            reject()
        }
    })
    // return Promise.reject()
}

function saveData(storeName: string, data: CacheStoreValue["data"]): Promise<any> {
    try {
        localStorage.setItem(storeName, JSON.stringify(data))
    } catch (e) {
        return Promise.reject(e)
    }
    return Promise.resolve()
}

export enum ElementType {
    TYPE,
    FIELD,
    METHOD,
    PARAMETER,
    CONSTRUCTOR,
    LOCAL_VARIABLE,
    ANNOTATION_TYPE,
    PACKAGE,
    TYPE_PARAMETER,
    TYPE_USE,
    MODULE,
    RECORD_COMPONENT
}

function Target(value: ElementType): MethodDecorator {
    return (target, propName, descriptor: PropertyDescriptor) => {

    }
}

function Documented(target: Object, propName: string, descriptor: PropertyDescriptor) {

}

enum RetentionPolicy {
    SOURCE,
    CLASS,
    RUNTIME
}

function Retention(value: RetentionPolicy = RetentionPolicy.CLASS): MethodDecorator {
    return (target, propName, descriptor: PropertyDescriptor) => {

    }
}

export default class Decorator {

    @Target(ElementType.METHOD)
    @Retention()
    @Documented
    static Override(target: Object, propName: string, descriptor: PropertyDescriptor) {
        let Class: Object | null
        try {
            Class = Reflect.getPrototypeOf(target)
        } catch (err) {
            throw new TypeError("override must be set in a class method")
        }
        if (Class && (Class as Function).name)
            throw new TypeError("override must be set in a class method which have a super class")
        if ((Class as Function).constructor.hasOwnProperty(propName))
            throw new TypeError("override must be set in a class method with already exist in super class")
    }

    //  参数默认值装饰器
    static DefaultArgs<T extends any[]>(...defaultArgs: T): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...realArgs: any[]) {
                const largerArgs = defaultArgs.length > realArgs.length ? defaultArgs : realArgs
                return originMethod.call(this, ...largerArgs.map((item, index) => {
                    const def = defaultArgs[index]
                    let real = realArgs[index] as any
                    if (def?.constructor === Object && real?.constructor === Object)
                        Object.keys(def).forEach(key => {
                            real[key] = real[key] ?? def[key]
                        })
                    return real ?? def
                }))
            }
        }
    }

    /**
     * 适配装饰器
     * @param adapter {function} 适配器
     */
    static UseAdapter<ReturnType extends any>(adapter: (result2: any) => ReturnType): MethodDecorator {
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

    static AddHost(_host: string = host.mainServer, attrs: AttrArray | string | unknown = undefined, adapter = convertDataUrl): MethodDecorator {
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
    static UseCache(storeName?: string, saveTime = initConfig.cacheValidityTime) {
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

    static Sign(apiUrl: string, adapter = getApiSign): MethodDecorator {
        return (target, propName, descriptor: PropertyDescriptor) => {
            const originMethod = descriptor.value
            descriptor.value = function (...[firstArg, otherArgs]: any[]) {
                firstArg.sign = adapter(apiUrl, firstArg)
                return originMethod.call(this, firstArg, ...otherArgs)
            }
        }
    }

    static ImplementsWithStatic<ConstructorType>() {
        return (constructor: ConstructorType) => {
        }
    }

    static IntIterator(startValue: number, length: number) {
        const int = {
            startValue,
            length,
            index: startValue,
            setIndex(index: number) {
                this.index = index
            },
            begin() {
                return this.startValue
            },
            end() {
                return this.length
            },
            toInt() {
                return this.index
            },
            toString() {
                return this.index.toString()
            },
            [Symbol.iterator]() {
                return {
                    next: (data: any) => {
                        this.index++
                        return {
                            value: this.index,
                            done: this.index >= this.length
                        }
                    },
                    return: () => {
                        this.index = this.startValue
                        return {
                            done: true,
                            value: undefined
                        }
                    }
                }
            }
        }
        Object.freeze(int)
        return int
    }

    _testOverride() {

    }

    static* generateWithLock<Args extends any[]>(lock: ILock, generatorFn: (...args: Args) => any, ...args: Args) {
        yield lock.acquire()
        try {
            yield* generatorFn(...args)
        } finally {
            lock.release()
        }
    }

    static withSpinLock(lock: ISpinLock) {
        return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
            const originalFn = descriptor.value;
    
            descriptor.value = async function (...args: any[]) {
                await lock.acquire()
                try {
                    return await originalFn.apply(this, args)
                } finally {
                    lock.release()
                }
            }
    
            return descriptor
        }
    }
}

export interface ILock {
    isLocked(): boolean
    acquire(): Promise<unknown>
    tryAcquire(): Promise<[null, Error | null]>
    release(): void
}

export interface ISpinLock extends ILock {
    setDelayTime(delayTime: number): this
}

export class SpinLock implements ISpinLock {
    private locked: boolean
    private delayTime: number

    constructor() {
        this.locked = false
        this.delayTime = 16
    }

    isLocked(): boolean {
        return this.locked
    }

    setDelayTime(delayTime: number) {
        this.delayTime = delayTime
        return this
    }

    acquire() {
        return new Promise((resolve, reject) => {
            const tryLock = () => {
                if (!this.locked) {
                    this.locked = true
                    resolve(void 0)
                } else {
                    setTimeout(tryLock, this.delayTime)
                }
            };
            tryLock()
        })
    }

    async tryAcquire(): Promise<[null, Error | null]> {
        try {
            await this.acquire()
            return [null, null]
        } catch (e) {
            return [null, e as Error]
        }
    }

    release() {
        this.locked = false
    }
}

const globalLock: ISpinLock = new SpinLock()

function testDecorator() {

    class TestOverride extends Decorator {
        @Decorator.Override
        testOverride() {
    
        }
    }

    new TestOverride()
    
    class Example {
        @Decorator.withSpinLock(globalLock)
        async criticalSection() {
            console.log("in")
            await new Promise((resolve) => setTimeout(resolve, 1000))
            console.log("out")
        }

        async* generatorSection() {
            yield new Promise((resolve) => setTimeout(resolve, 1000))
        }
    }
    const example = new Example()

    example.criticalSection()
    example.criticalSection(); // lock and wait

    (async () => {
        const generator = Decorator.generateWithLock(globalLock, example.generatorSection.bind(example))
        for await (const value of generator) {
            console.log("Generator waiting...")
        }
    })()
}