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
        cwd: "./desktop"
    }, err => {
        if (err) throw err
    })
    desktopProcess.stdout.on("data", data => {
        console.log(`Desktop: ${data}`)
    })
}

const gatewayProcess = exec("npm run start:dev", {
    cwd: "./gateway"
}, err => {
    if (err) throw err
})
let timer = null
gatewayProcess.stdout.on("data", data => {
    console.log(`Gateway: ${data}`)
    clearTimeout(timer)
    timer = setTimeout(() => {
        !clientProcess && startClientDev()
        !desktopProcess && startDesktopDev()
    }, 1500)
})
