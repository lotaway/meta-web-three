const {app, BrowserWindow, ipcMain} = require("electron")
const remote = require("@electron/remote/main")
const path = require("path")
const fs = require("fs")

const isDev = global.process.env.NODE_ENV === "development"
const isLinux = process.platform === "linux"
const isWin = process.platform === "win32"
const isMac = process.platform === "darwin"
let mainWindow = null

async function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            devTools: isDev,
            nodeIntegration: true,
            contextIsolation: false,
            preload: path.join(__dirname, "./preload.js")
        }
    })
    remote.initialize()
    remote.enable(mainWindow.webContents)
    // isDev && mainWindow.webContents.openDevTools()
    if (isDev) {
        await mainWindow.loadURL("http://localhost:30002")
    } else {
        await mainWindow.loadFile("dist/index.html")
    }
}

app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0 || mainWindow === null)
        void createWindow().catch(err => {
            console.log("重新创建窗口失败" + JSON.stringify(err))
        })
})
app.on("window-all-closed", () => {
    !isMac && app.quit()
})
void app.whenReady().then(() => {
})

void app.whenReady().then(() => {
    function init() {
        void createWindow().catch(err => {
            console.log("创建窗口失败：" + JSON.stringify(err))
        })
    }

    // 延迟解决莫名其妙的ready未完成问题：https://github.com/electron/electron/issues/16809
    isLinux ? setTimeout(init, 300) : init()
})

class Handler {

    static readFileInDirectory(directory) {
        const names = getIncludeFiles(directory)
        const paths = filename2path(names, directory)
        return {
            names,
            paths
        }
    }
}


function getIncludeFiles(path) {
    const dirents = fs.readdirSync(path, {
        withFileTypes: true
    })
    if (dirents.length === 0) return []
    return dirents.filter((item, index) => item.isFile()).map(item => item.name)
}

function filename2path(filenames, prevFix) {
    return filenames.map(filename => path.join(prevFix, filename))
}

ipcMain.handle("readFileInDirectory", filePath => Handler.readFileInDirectory(filePath))
