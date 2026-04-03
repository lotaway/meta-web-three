const fs = require('node:fs')
const path = require('node:path')
const dotenv = require("dotenv")
dotenv.config({path: path.resolve(__dirname, '../.env.development')});
dotenv.config({path: path.resolve(__dirname, '../.env.local')});

// Java DIR
const javaEnumsRootDir = path.resolve(__dirname, process.env.BACKEND_API_ROOT_DIR || '../../backend-api')
// TypeScript output DIR
const tsEnumsRootDir = path.resolve(__dirname, '../src/generated/')

function processJavaEnumFiles(dir) {
    let hasValidEnum = false

    fs.readdirSync(dir).forEach(file => {
        const fullPath = path.join(dir, file)
        const relativePath = path.relative(javaEnumsRootDir, fullPath)
        const outputDir = path.join(tsEnumsRootDir, path.dirname(relativePath))

        if (fs.lstatSync(fullPath).isDirectory()) {
            const subDirHasValidEnum = processJavaEnumFiles(fullPath)
            hasValidEnum = hasValidEnum || subDirHasValidEnum
        } else if (path.extname(file) === '.java') {
            const javaEnum = fs.readFileSync(fullPath, 'utf8')
            const tsEnum = convertJavaEnumToTs(javaEnum)

            if (tsEnum) {
                if (!fs.existsSync(outputDir)) {
                    fs.mkdirSync(outputDir, {recursive: true})
                }
                const tsFileName = path.basename(file, '.java') + '.ts'
                fs.writeFileSync(path.join(outputDir, tsFileName), tsEnum)
                console.log(`Converted ${relativePath} to ${path.join(outputDir, tsFileName)}`)
                hasValidEnum = true
            }
        }
    })

    return hasValidEnum
}

function convertJavaEnumToTs(javaEnum) {
    const enumNameMatch = javaEnum.match(/public enum (\w+)/)
    const enumValuesMatch = javaEnum.match(/{([\s\S]+?)}/)

    if (!enumNameMatch || !enumValuesMatch) {
        return null
    }

    const enumName = enumNameMatch[1]
    const enumValuesBlock = enumValuesMatch[1].trim()
    const enumLines = enumValuesBlock.split('\n').map(line => line.trim())

    let tsEnum = `export enum ${enumName} {\n`

    enumLines.forEach(line => {
        if (line.startsWith('//')) {
            tsEnum += `    ${line}\n`
        } else {
            const valueMatch = line.match(/^(\w+)\(([^)]+)\)[,]?/)
            if (valueMatch) {
                const valueName = valueMatch[1].trim()
                const params = valueMatch[2].split(',').map(param => param.trim().replace(/^"(.*)"$/, '$1'))
                if (params.length === 1) {
                    let paramValue = params[0]
                    if (/^(-)?\d+[LFD]/.test(paramValue)) {
                        paramValue = paramValue.slice(0, -1);
                    }
                    tsEnum += `    ${valueName} = ${paramValue},\n`
                } else if (params.length === 2) {
                    const key = params[0]
                    const description = params[1].replace(/"/g, '')
                    tsEnum += `    // ${description}\n`
                    tsEnum += `    ${valueName} = "${key}",\n`
                }
            }
        }
    })

    tsEnum += `}\n`

    return tsEnum
}

if (processJavaEnumFiles(javaEnumsRootDir)) {
    console.log('TypeScript enums generated successfully.')
} else {
    console.log('No valid enums found. No files or directories were generated.')
}
