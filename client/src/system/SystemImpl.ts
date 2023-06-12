import Decorator, {obj2FormData} from "../utils/decorator"
import * as IFBase from "../services/IFBase"
import {IBaseMapperRequestOptions} from "../services/IFBase"

@Decorator.ImplementsWithStatic<IFBase.ISystemWithStatic>()
export default class SystemImpl {
    private static instance: SystemImpl | undefined
    static getInstance() {
        if (this.instance)
            return this.instance
        this.instance = new SystemImpl()
        return this.instance
    }
    uploadFile(apiUrl: string, file: File, options: IFBase.UploadFileArgs) {
        const formData = new FormData()
        formData.append("file", file)
        let headers = options.headers || {}
        headers['Content-Type'] = 'multipart/form-data'
        return fetch(apiUrl, {
            method: options.method || "POST",
            body: formData,
            headers
        }).then(response => response.ok ? response.json() : Promise.reject(response))
    }

    async request<responseData = any>(apiUrl: string, data: object, options?: IBaseMapperRequestOptions) {
        return await fetch(apiUrl, {
            method: options?.method || "POST",
            body: obj2FormData(data),
            signal: options?.signal
        }).then(response => response.ok ? (response.json() as Promise<responseData>) : Promise.reject(response))
    }
}
