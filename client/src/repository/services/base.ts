export interface IApiProvider<Arguments = any, ResponseData = any> {

    abortController: AbortController | undefined

    start(args: Arguments): Promise<ResponseData>

    stop(): boolean
}

export interface IApiProviderStatic<InstanceType> {
    new(rpc: ISystem, options?: IBaseProviderOptions): InstanceType
}

export class BaseProvider {

    abortController: AbortController | undefined = undefined

    constructor(protected readonly rpc: ISystem, protected options: IBaseProviderOptions = {}) {
    }

    init() {
        this.abortController = new AbortController()
    }

    stop() {
        if (!this.abortController)
            return false
        this.abortController.abort("call stop")
        this.abortController = undefined
        return true
    }
}

export function ServiceWrapper() {

}

export class MapperWrapper<DefaultArgs = any> {
    constructor(private readonly systemImpl: ISystem, private readonly defaultArgs?: Partial<DefaultArgs>) {
    }

    get system(): ISystem {
        return this.systemImpl
    }

    start<Args extends DefaultArgs, ResponseData>(Mapper: IApiProviderStatic<IApiProvider<Args, ResponseData>>, args: Omit<Args, keyof DefaultArgs> & Partial<DefaultArgs>) {
        const mapper = new Mapper(this.systemImpl)
        return mapper.start({
            ...this.defaultArgs,
            ...args
        } as Args)
    }
}
