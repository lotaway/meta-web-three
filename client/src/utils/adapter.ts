//  参数默认值装饰器
export function setDefaultArgs(...defaultArgs: any[]): MethodDecorator {
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
export function useAdapter<ReturnType>(adapter: (result2: any) => ReturnType): MethodDecorator {
    return (target, propName, descriptor: PropertyDescriptor) => {
        const originMethod = descriptor.value
        descriptor.value = function (...params: any[]) {
            if (originMethod === "Promise")
                return originMethod.call(this, ...params).then(adapter)
            else
                return adapter(originMethod.call(this, ...params))
        }
    }
}
