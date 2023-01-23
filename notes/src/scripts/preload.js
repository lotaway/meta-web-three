const {contextBridge, ipcMain} = require("electron")

contextBridge.exposeInMainWorld("desktop", {
    "emitMain": (event, data) => ipcMain.emit(event, data)
})

window.addEventListener("DOMContentLoaded", () => {

})
