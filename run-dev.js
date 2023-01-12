const {exec} = require("child_process")
exec("npm run dev", {
    cwd: "./client"
}, (err, stdout, stderr) => {
    if (err) throw err
    console.log(`client stdout:${stdout.toString()}`)
    console.log(`client stderr:${stderr.toString()}`)
})
exec("npm run start:dev", {
    cwd: "./server"
}, (err, stdout, stderr) => {
    if (err) throw err
    console.log(`server stdout: ${stdout.toString()}`)
    console.error(`server stdout:${stderr.toString()}`)
})