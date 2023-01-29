export {}

type ApiKey = string | number

declare global {
    interface Window {
        "desktop": {
            [apiKey: ApiKey]: (...params: any) => any
        }
    }
}
