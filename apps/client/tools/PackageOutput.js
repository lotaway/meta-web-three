const fs = require('node:fs')
const path = require('node:path')
const { execSync } = require('node:child_process')
const archiver = require('archiver')

const currentDir = process.cwd()

function getOutputDir() {
    const NEXT_DEFAULT_PUTOUT = 'out'
    const configPath = path.join(process.cwd(), 'next.config.js')
    if (fs.existsSync(configPath)) {
        const config = require(configPath)
        console.log(`getOutputDir:${config.distDir}`)
        return config.distDir ?? NEXT_DEFAULT_PUTOUT
    }
    console.log(`getOutputDir:${NEXT_DEFAULT_PUTOUT}`)
    return NEXT_DEFAULT_PUTOUT
}

const folderName = getOutputDir()
const folderPath = path.join(currentDir, folderName)

function getCurrentBranch() {
    return execSync('git rev-parse --abbrev-ref HEAD').toString().trim()
}

function getCurrentCommitHash() {
    return execSync('git rev-parse HEAD').toString().trim()
}

function getCurrentTimestamp() {
    return Math.floor(Date.now() / 1000)
}

async function createZip() {
    let branch = getCurrentBranch().split("/")
    branch = branch[branch.length - 1].trim()
    const commitHash = getCurrentCommitHash().slice(-8)
    const timestamp = getCurrentTimestamp()
    const zipFileName = `${path.basename(currentDir)}-${folderName}-${branch}-${commitHash}.zip`
    const outputPath = path.join(currentDir, zipFileName)

    const output = fs.createWriteStream(outputPath, {
        flags: "w"
    })
    const archive = archiver('zip', {
        zlib: { level: 9 }
    })

    output.on('close', () => {
        console.log(`Zip package created: ${zipFileName} (${archive.pointer()} bytes)`)
    })

    archive.on('error', (err) => {
        throw err
    })

    archive.pipe(output)
    archive.directory(folderPath, folderName)
    await archive.finalize()
}

if (fs.existsSync(folderPath)) {
    createZip().finally(() => {
        console.log('Finish.')
    })
} else {
    console.error(`directory ${folderName} not exists.`)
}