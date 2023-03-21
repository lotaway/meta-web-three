import Decorator, {obj2FormData} from "../utils/decorator"
import * as IFBase from "./IFBase"

@Decorator.implementsWithStatic<IFBase.IFBaseServiceWithStatic>()
export default class BaseService {
    static uploadFile(apiUrl: string, file: File, options: IFBase.UploadFileArgs) {
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

    static async request<responseData extends any>(apiUrl: string, data: object, options?: { dataType?: string, method?: string }) {
        return await fetch(apiUrl, {
            method: options?.method || "POST",
            body: obj2FormData(data)
        }).then(response => response.ok ? (response.json() as Promise<responseData>) : Promise.reject(response))
    }
}
