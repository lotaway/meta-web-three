interface Swing {
    com(type: number): void
}

function useSwing(swing: Swing | Swing["com"]) {

}

function createUseSwing() {
    useSwing(new class implements Swing {
        com(type: number): void {

        }
    });
    useSwing(type => {
        type++;
    })
}

export interface UploadFileArgs {
    method?: string
    headers?: {
        [key: string]: string
    }
}

export interface IBaseMapperRequestOptions {
    dataType?: string,
    method?: string,
    signal?: AbortSignal
}

export interface ISystem {
    uploadFile(apiUrl: string, file: File, options: UploadFileArgs): Promise<any>;

    // stopFile(): boolean
    request<responseData extends any>(apiUrl: string, data: object, options?: IBaseMapperRequestOptions): Promise<responseData>
}

export interface ISystemWithStatic<T extends ISystem = ISystem> {
    new(...args: any[]): T;
}
