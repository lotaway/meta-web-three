const {app, BrowserWindow} = require("electron")
const remote = require("@electron/remote/main")
const path = require("path")
const isDev = process.env.NODE_ENV === "development"
const isLinux = process.platform === "linux"
const isWin = process.platform === "win32"
const isMac = process.platform === "darwin"
let mainWindow = null

async function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            devTools: true,
            nodeIntegration: true,
            preload: path.join(__dirname, "./src/scripts/preload.js")
        }
    })
    remote.initialize()
    remote.enable(mainWindow.webContents)
    mainWindow.webContents.openDevTools()
    if (isDev) {
        await mainWindow.loadURL("http://localhost:30002")
    } else {
        await mainWindow.loadFile("dist/index.html")
    }
}

app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0 || mainWindow === null)
        createWindow().catch(err => {
            console.log("重新创建窗口失败" + JSON.stringify(err))
        })
})
app.on("window-all-closed", () => {
    !isMac && app.quit()
})
app.whenReady().then(() => {
    createWindow().catch(err => {
        console.log("创建窗口失败：" + JSON.stringify(err))
    })
})
