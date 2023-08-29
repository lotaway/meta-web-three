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

startClientDev()