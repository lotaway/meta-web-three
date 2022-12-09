namespace system {
    export type IP = string

    export interface Info {
        ip: IP
        name: string
        version: number
        type: string
        storage: number
    }

    export function getIP() {
        return "It's ip";
    }
}