import {IBaseMapperRequestOptions, ISystem} from "./IFBase"
import SystemImpl from "../system/SystemImpl";
import Decorator from "../utils/decorator";

export interface IApiMapper<Arguments = any, ResponseData = any> {

    abortController: AbortController | undefined

    start(args: Arguments): Promise<ResponseData>

    stop(): boolean
}

export interface IApiMapperStatic<InstanceType> {
    new(rpc: ISystem, options?: IBaseMapperRequestOptions): InstanceType
}

export class BaseMapper {

    abortController: AbortController | undefined = undefined
    constructor(protected readonly rpc: SystemImpl, protected options: IBaseMapperRequestOptions = {}) {
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
