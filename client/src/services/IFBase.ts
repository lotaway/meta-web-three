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

export interface IFBaseService {

}

export interface IFBaseServiceWithStatic<T = IFBaseService> {
    new(...args: any[]): T;

    uploadFile(apiUrl: string, file: File, options: UploadFileArgs): Promise<any>;

    // stopFile(): boolean
}
