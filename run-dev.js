const {exec} = require("child_process")
let clientProcess = null
let desktopProcess = null

function startClientDev() {
    clientProcess = exec("npm run dev", {
        cwd: "./client"
    }, err => {
        if (err) throw err
    })
    clientProcess.stdout.on("data", data => {
        console.log(`Client: ${data}`)
    })
}

function startDesktopDev() {
    desktopProcess = exec("npm run dev", {
        cwd: "./notes"
    }, err => {
        if (err) throw err
    })
    desktopProcess.stdout.on("data", data => {
        console.log(`Notes: ${data}`)
    })
}

const serverProcess = exec("npm run start:dev", {
    cwd: "./server"
}, err => {
    if (err) throw err
})
let timer = null
serverProcess.stdout.on("data", data => {
    console.log(`Server: ${data}`)
    clearTimeout(timer)
    timer = setTimeout(() => {
        !clientProcess && startClientDev()
        !desktopProcess && startDesktopDev()
    }, 1500)
})
