const {contextBridge, ipcMain, ipcRenderer, BrowserWindow} = require("electron")

type ReceiveFn = (...args: any) => void

const desktopFn = {
    send: (channel: string, data: any) => {
        ipcRenderer.invoke(channel, data).catch(e => console.log(e))
    },
    receive: (channel: string, func: ReceiveFn) => {
        console.log("preload-receive called. args: ")
        ipcRenderer.on(channel, (event, ...args) => func(...args))
    },
    ipcSendTo: (window_id: number, channel: string, ...arg: any) => {
        ipcRenderer.sendTo(window_id, channel, arg)
    },
    ipcSend: (channel: string, ...arg: any) => {
        ipcRenderer.send(channel, arg)
    },
    ipcSendSync: (channel: string, ...arg: any) => {
        return ipcRenderer.sendSync(channel, arg)
    },
    ipcOn: (channel: string, listener: (event: any, ...arg: any) => void) => {
        ipcRenderer.on(channel, listener)
    },
    ipcOnce: (channel: string, listener: (event: any, ...arg: any) => void) => {
        ipcRenderer.once(channel, listener)
    },
    ipcRemoveListener: (channel: string, listener: (event: any, ...arg: any) =>
        void) => {
        ipcRenderer.removeListener(channel, listener)
    },
    ipcRemoveAllListeners: (channel: string) => {
        ipcRenderer.removeAllListeners(channel)
    }
}
if (process.contextIsolated) {
    contextBridge.exposeInMainWorld("desktop", desktopFn)
} else {
    window.desktop = desktopFn
}
window.addEventListener("DOMContentLoaded", () => {

})
export {}
