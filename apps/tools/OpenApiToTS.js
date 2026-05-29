// OpenAPI to TypeScript code generator (shared version)
// Usage: node apps/tools/OpenApiToTS.js
//
// Config via environment variables:
//   OPENAPI_ENV_FILES        - comma-separated env files (default: .env)
//   OPENAPI_API_HOST_ENV     - env var name for API host (default: VITE_BASE_SERVER_URL)
//   OPENAPI_API_DOC_HOST_ENV - env var name for API doc host (default: same as API_HOST)
//   OPENAPI_OUTPUT_DIR       - output dir relative to cwd (default: src/generated/api)
//   OPENAPI_RUNTIME_ENV_VAR  - runtime env expression (default: import.meta.env.VITE_BASE_SERVER_URL)
//   OPENAPI_TIMEOUT          - timeout in ms (default: 180000)
//   OPENAPI_USE_NPX          - set "true" to use npx prefix (default: false)
//   OPENAPI_DEFAULT_HOST     - fallback host (default: http://localhost:10081)

const fs = require("node:fs")
const https = require("node:https")
const http = require("node:http")
const { spawn } = require("node:child_process")
const path = require("node:path")
const dotenv = require("dotenv")

const projectRoot = process.cwd()

function readConfig() {
  const envFiles = (process.env.OPENAPI_ENV_FILES || ".env").split(",")
  for (const file of envFiles) {
    const filePath = path.resolve(projectRoot, file.trim())
    if (fs.existsSync(filePath)) {
      dotenv.config({ path: filePath })
    }
  }

  const apiHostEnv = process.env.OPENAPI_API_HOST_ENV || "VITE_BASE_SERVER_URL"
  const apiDocHostEnv =
    process.env.OPENAPI_API_DOC_HOST_ENV || apiHostEnv
  const defaultHost = process.env.OPENAPI_DEFAULT_HOST || "http://localhost:10081"

  return {
    apiHost: process.env[apiHostEnv] || defaultHost,
    apiDocHost: process.env[apiDocHostEnv] || process.env[apiHostEnv] || defaultHost,
    outputDir: process.env.OPENAPI_OUTPUT_DIR || "src/generated/api",
    runtimeEnvVar:
      process.env.OPENAPI_RUNTIME_ENV_VAR || "import.meta.env.VITE_BASE_SERVER_URL",
    timeout: parseInt(process.env.OPENAPI_TIMEOUT || "180000", 10),
    useNpx: process.env.OPENAPI_USE_NPX === "true",
    apiHostEnvName: apiHostEnv,
  }
}

function getApiDocsUrl(config) {
  return `${config.apiDocHost}/meta/v3/api-docs`
}

function downloadOpenApiDocs(config) {
  const apiDocsUrl = getApiDocsUrl(config)
  const outputPath = path.resolve(projectRoot, config.outputDir, "openapi.json")

  return new Promise((resolve, reject) => {
    const clientReq = (apiDocsUrl.startsWith("https") ? https : http).get(
      apiDocsUrl,
      (clientRes) => {
        let chunks = ""
        clientRes.on("data", (chunk) => {
          chunks += chunk
        })
        clientRes.on("end", async () => {
          const dir = path.dirname(outputPath)
          await fs.promises.mkdir(dir, { recursive: true })
          const requestUrl = new URL(apiDocsUrl)
          const configJSON = JSON.parse(chunks)
          configJSON.servers = configJSON.servers?.map((item) => {
            try {
              const urlStr = item.url === "/" ? requestUrl.origin : item.url
              const url = new URL(urlStr)
              if (url.host === requestUrl.host || item.url === "/") {
                item.url = requestUrl.origin
              }
            } catch {
              item.url = requestUrl.origin
            }
            return item
          })
          await fs.promises.writeFile(outputPath, JSON.stringify(configJSON, null, 2))
          console.log(`OpenAPI docs downloaded and saved to ${outputPath}`)
          resolve(outputPath)
        })
        clientRes.on("error", (err) => reject(err))
        clientRes.on("close", () => console.log("Connection closed"))
      }
    )
    clientReq.on("error", (err) => reject(err))
  })
}

function generateCode(config, inputPath) {
  const output = path.resolve(projectRoot, config.outputDir)
  const args = [
    "generate",
    "-i", inputPath,
    "-g", "typescript-fetch",
    "-o", output,
    "--additional-properties=useSingleRequestParameter=true",
    "--skip-validate-spec",
  ]

  return new Promise((resolve, reject) => {
    const child = config.useNpx
      ? spawn("npx", ["openapi-generator", ...args], {
          stdio: "inherit",
          cwd: projectRoot,
        })
      : spawn("openapi-generator", args, {
          stdio: "inherit",
          cwd: projectRoot,
        })

    const timeout = setTimeout(() => {
      child.kill()
      reject(new Error("Time out"))
    }, config.timeout)

    child.on("close", (code) => {
      clearTimeout(timeout)
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`Generator exited with code: ${code}`))
      }
    })

    child.on("error", (err) => {
      clearTimeout(timeout)
      reject(err)
    })
  })
}

async function readOpenApiConfigFile(config) {
  const outputPath = path.resolve(projectRoot, config.outputDir, "openapi.json")
  const chunks = await fs.promises.readFile(outputPath, "utf-8")
  return JSON.parse(chunks)
}

async function customizeCode(config) {
  const [openApiConfig, runtimeFileStr] = await Promise.all([
    readOpenApiConfigFile(config),
    fs.promises.readFile(
      path.resolve(projectRoot, config.outputDir, "runtime.ts"),
      "utf-8"
    ),
  ])

  try {
    const apiDocsUrl = getApiDocsUrl(config)
    const urlIns = new URL(apiDocsUrl)
    const url = openApiConfig.servers?.find((item) => {
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
      `\`\${${config.runtimeEnvVar} || "${urlIns.origin}"}${serverUrl.pathname === "/" ? "" : serverUrl.pathname}\``
    )
    await fs.promises.writeFile(
      path.resolve(projectRoot, config.outputDir, "runtime.ts"),
      newStr
    )
  } catch (error) {
    console.log("Skipping server URL customization:", error.message)
  }
}

async function main() {
  const config = readConfig()
  const apiDocsUrl = getApiDocsUrl(config)

  try {
    console.log("Start downloadOpenApiDocs...", apiDocsUrl)
    const downloadedFilePath = await downloadOpenApiDocs(config)
    console.log("End downloadOpenApiDocs...")

    console.log("Start generateCode...")
    await generateCode(config, downloadedFilePath)
    console.log("generateCode done.")

    console.log("Start customize code...")
    await customizeCode(config)
    console.log("Customize code done.")
  } catch (error) {
    console.error("generateCode error:", error)
  }
}

main().finally(() => process.exit(0))
