import Decorator, {obj2FormData} from "../commons/utils/support"
import Logger from "./Logger"

@Decorator.ImplementsWithStatic<ISystemWithStatic>()
export default class System {
    private static instance: ISystem | undefined

    constructor(protected readonly loggerImpl: ILogger) {
    }

    static getInstance() {
        if (this.instance)
            return this.instance
        this.instance = new System(Logger.getInstance())
        return this.instance
    }

    async uploadFile(apiUrl: string, file: File, options: UploadFileArgs) {
        const formData = new FormData()
        formData.append("file", file)
        let headers = options.headers || {}
        headers['Content-Type'] = 'multipart/form-data'
        return await fetch(apiUrl, {
            method: options.method || "POST",
            body: formData,
            headers
        }).then(response => response.ok ? response.json() : Promise.reject(response))
    }

    async request<responseData = any>(apiUrl: string, data: object, options?: IBaseProviderOptions) {
        return await fetch(apiUrl, {
            method: options?.method || "POST",
            body: obj2FormData(data),
            signal: options?.signal
        }).then(response => response.ok ? (response.json() as Promise<responseData>) : Promise.reject(response))
    }

    print(message: string) {
        this.loggerImpl.output(message)
    }

    language() {
        return navigator.language
    }
}
