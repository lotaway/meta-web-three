// https://openapi-generator.tech/docs/installation/
// openapi-generator-cli generate -i ./src/generated/api/openapi.json -g typescript-fetch
const fs = require("node:fs")
const https = require("node:https")
const http = require("node:http")
const { spawn } = require("cross-spawn")
const { execSync } = require("child_process")
const path = require("node:path")
const dotenv = require("dotenv")
dotenv.config({ path: path.resolve(__dirname, "../.env.development") })
dotenv.config({ path: path.resolve(__dirname, "../.env.local") })

// API Doc url
const apiHost = process.env.NEXT_PUBLIC_BACK_API_HOST
const apiDocHost = process.env.NEXT_PUBLIC_BACK_API_DOC_HOST ?? apiHost ?? ""
const apiDocsUrl = `${apiDocHost}/pump/v3/api-docs`
// TypeScript output DIR
const outputDir = "../src/generated/api"
const outputPath = path.join(__dirname, outputDir, "./openapi.json")

function downloadOpenApiDocs() {
    return new Promise((resolve, reject) => {
        const clientReq = (apiDocsUrl.startsWith("https") ? https : http).get(
            apiDocsUrl,
            (clientRes) => {
                let chunks = ""
                const target = clientRes
                target.on("data", (chunk) => {
                    chunks += chunk
                })
                target.on("end", async () => {
                    // const data = Buffer.concat(chunks)
                    const dir = path.dirname(outputPath)
                    await fs.promises.mkdir(dir, { recursive: true })
                    const request_url = new URL(apiDocsUrl)
                    const configJSON = JSON.parse(chunks)
                    configJSON.servers = configJSON.servers?.map((item) => {
                        try {
                            const urlStr = item.url === '/' ? request_url.origin : item.url
                            const url = new URL(urlStr)
                            if (url.host === request_url.host || item.url === '/') {
                                item.url = request_url.origin
                            }
                        } catch (e) {
                            item.url = request_url.origin
                        }
                        return item
                    })
                    await fs.promises.writeFile(
                        outputPath,
                        JSON.stringify(configJSON)
                    )
                    console.log(
                        `OpenAPI docs downloaded and saved to ${outputPath}`
                    )
                    resolve(outputPath)
                })
                target.on("error", (err) => {
                    reject(err)
                })
                target.on("close", () => {
                    console.log("Connection closed")
                })
            }
        )
    })
}

async function generateCode(inputPath) {
    const output = path.resolve(__dirname, outputDir)
    const args = [
        'generate',
        '-i', inputPath,
        '-g', 'typescript-fetch',
        '-o', output,
        '--additional-properties=useSingleRequestParameter=true',
        '--skip-validate-spec',
    ]

    return new Promise((resolve, reject) => {
        const child = spawn('openapi-generator', args, {
            stdio: 'inherit',
            cwd: path.resolve(__dirname, '../')
        })

        const timeout = setTimeout(() => {
            child.kill();
            reject(new Error("Time out"));
        }, 120 * 1000);

        child.on('close', (code) => {
            clearTimeout(timeout);
            if (code === 0) {
                resolve()
            } else {
                reject(new Error(`Generator exited with code: ${code}`))
            }
        })
    })
}

function runCommand(command, args) {
    return new Promise((resolve, reject) => {
        const options = {
            stdio: "inherit",
            cwd: path.resolve(__dirname, "../"),
        }
        const proc = spawn(command, args, options)
        proc.stdout?.on?.("data", (data) => {
            console.log(`output: ${data}`)
        })
        proc.stderr?.on?.("data", (data) => {
            reject(`error: ${data}`)
        })
        proc.on("close", (code) => {
            if (code === 0) {
                resolve()
            } else {
                reject(`process exited with code: ${code}`)
            }
        })
        proc.on("error", (error) => {
            reject(error)
        })
    })
}

async function readOpenApiConfigFile() {
    const chunks = await fs.promises.readFile(outputPath, "utf-8")
    return JSON.parse(chunks)
}

async function customizeCode() {
    const [config, runtimeFileStr] = await Promise.all([
        readOpenApiConfigFile(),
        fs.promises.readFile(
            path.join(__dirname, outputDir, "runtime.ts"),
            "utf-8"
        ),
    ])
    try {
        const urlIns = new URL(apiDocsUrl)
        const url = config.servers?.find((item) => {
            try {
                return item.url.includes(urlIns.origin)
            } catch {
                return false
            }
        })?.url
        if (!url) {
            console.log("No server URL customization needed.")
            return
        }
        const serverUrl = new URL(url)
        const newStr = runtimeFileStr.replace(
            `"${url}"`,
            `\`\${process.env.NEXT_PUBLIC_BACK_API_HOST ?? "${urlIns.origin}"}${serverUrl.pathname === '/' ? '' : serverUrl.pathname}\``
        )
        await fs.promises.writeFile(
            path.join(__dirname, outputDir, "runtime.ts"),
            newStr
        )
    } catch (error) {
        console.log("Skipping server URL customization:", error.message)
    }
}

async function main() {
    try {
        console.log("Start downloadOpenApiDocs...", apiDocsUrl)
        const downloadedFilePath = await downloadOpenApiDocs()
        console.log("End downloadOpenApiDocs...")
        console.log("Start generateCode...")
        await generateCode(path.resolve(__dirname, downloadedFilePath))
        console.log("generateCode done.")
        console.log("Start customize code...")
        await customizeCode()
        console.log("Customize code done.")
    } catch (error) {
        console.error("generateCode error:", error)
    }
}

main().finally(() => {
    process.exit(0)
})
