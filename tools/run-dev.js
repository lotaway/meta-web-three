const {exec} = require("child_process")
let clientProcess = null
let desktopProcess = null

function startClientDev() {
    clientProcess = exec("npm run dev", {
        cwd: "../client"
    }, err => {
        if (err) throw err
    })
    clientProcess.stdout.on("data", data => {
        console.log(`client: ${data}`)
    })
}

function startDesktopDev() {
    desktopProcess = exec("npm run dev", {
        cwd: "../system-management"
    }, err => {
        if (err) throw err
    })
    desktopProcess.stdout.on("data", data => {
        console.log(`system-management: ${data}`)
    })
}

const gatewayProcess = exec("npm run start:dev", {
    cwd: "../gateway"
}, err => {
    if (err) throw err
})
let timer = null
gatewayProcess.stdout.on("data", data => {
    console.log(`gateway: ${data}`)
    clearTimeout(timer)
    timer = setTimeout(() => {
        !clientProcess && startClientDev()
        !desktopProcess && startDesktopDev()
    }, 1500)
})
