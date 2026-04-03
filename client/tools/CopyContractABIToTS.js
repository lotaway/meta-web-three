const fs = require('node:fs')
const path = require('node:path')
const solc = require('solc')
const {execSync} = require('child_process')
const dotenv = require("dotenv")
dotenv.config({ path: path.resolve(__dirname, '../.env.development') });
dotenv.config({ path: path.resolve(__dirname, '../.env.local') });

const CONTRACT_PROJECT_PATH = path.resolve(__dirname, process.env.CONTRACT_ROOT_DIR ?? '../../contract')
const CONTRACTS_PATH = path.resolve(__dirname, CONTRACT_PROJECT_PATH, 'artifacts/contracts')
const TARGET_PATH = path.resolve(__dirname, '../src/generated/contract')

function copyDirectory(source, target) {
    if (!fs.existsSync(target)) {
        fs.mkdirSync(target, { recursive: true })
    }

    const files = fs.readdirSync(source)

    files.forEach(file => {
        const sourcePath = path.join(source, file)
        const targetPath = path.join(target, file)

        if (fs.lstatSync(sourcePath).isDirectory()) {
            copyDirectory(sourcePath, targetPath)
        } else {
            fs.copyFileSync(sourcePath, targetPath)
        }
    })
}

function copyAbiFiles() {
    const targetDir = path.resolve(__dirname, TARGET_PATH)
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true })
    }
    const files = fs.readdirSync(CONTRACTS_PATH, { recursive: true })

    files.forEach(file => {
        if (file.endsWith('.json') && !file.endsWith('.dbg.json')) {
            const filePath = path.join(CONTRACTS_PATH, file)
            const content = JSON.parse(fs.readFileSync(filePath, 'utf8'))
            const contractName = path.basename(file, '.json')
            const targetAbiFile = path.join(targetDir, `${contractName}Abi.ts`)
            const abiContent = `// Generated from ${file}\n\nexport const ${contractName}Abi = ${JSON.stringify(content.abi, null, 2)} as const;\n`
            fs.writeFileSync(targetAbiFile, abiContent)
            const targetJSONFile = path.join(targetDir, `${contractName}.json`)
            fs.writeFileSync(targetJSONFile, JSON.stringify(content))
            console.log(`Generated ${targetAbiFile} and ${targetJSONFile}`)
        }
    })
}

function generateContractTypeFiles() {
    console.log("generateContractABIAndTypeFiles start...")
    const targetDir = path.resolve(__dirname, TARGET_PATH)
    if (process.platform === 'win32') {
        execSync(`set NODE_ENV=production && npx typechain --target ethers-v6 --out-dir ${targetDir}/types ${targetDir}/*.json`, {
            shell: true
        })
    } else {
        execSync(`NODE_ENV=production npx typechain --target ethers-v6 --out-dir ${targetDir}/types ${targetDir}/*.json`)
    }
    console.log("generateContractABIAndTypeFiles done.")
}

// interface EnumDefinition {
//     name: string
//     members: string[]
//     documentation ?: string
// }

/**
 * 
 * @param {*} ast 
 * @returns EnumDefinition[]
 */
function findEnums(ast) {
    const enums = []

    function visit(node) {
        if (node.nodeType === 'EnumDefinition') {
            const enumDef = {
                name: node.name,
                members: node.members.map((member) => member.name),
                documentation: node.documentation?.text
            }
            enums.push(enumDef)
        }
        for (const key in node) {
            if (typeof node[key] === 'object' && node[key] !== null) {
                visit(node[key])
            }
        }
    }

    visit(ast)
    return enums
}

/**
 * 
 * @param {EnumDefinition} enumDef 
 * @returns {string}
 */
function generateEnumTS(enumDef) {
    const members = enumDef.members
        .map((member, index) => `  ${member} = ${index}`)
        .join(',\n')

    return `export enum ${enumDef.name} {
${members}
}`
}

/**
 * 
 * @param {string} solidityFilePath 
 * @param {string} outputPath 
 * @returns {void}
 */
async function generateEnumsFromSolidity(solidityFilePath, outputPath) {
    const source = fs.readFileSync(solidityFilePath, 'utf8')
    const input = {
        language: 'Solidity',
        sources: {
            [path.basename(solidityFilePath)]: {
                content: source
            }
        },
        settings: {
            outputSelection: {
                '*': {
                    '*': ['ast']
                }
            }
        }
    }
    const output = JSON.parse(solc.compile(JSON.stringify(input)))
    console.log("output", output.sources)
    const ast = output.sources[path.basename(solidityFilePath)].ast
    const enums = findEnums(ast)
    const tsCode = enums
        .map(enumDef => generateEnumTS(enumDef))
        .join('\n\n')
    fs.writeFileSync(outputPath, tsCode)
    console.log(`Generated enums in ${outputPath}`)
}

async function generateExtra(contractSrcPath) {
    if (!fs.existsSync(TARGET_PATH)) {
        fs.mkdirSync(TARGET_PATH, { recursive: true })
    }
    const entries = fs.readdirSync(contractSrcPath)
    entries
        .filter(file => file.endsWith('.sol'))
        .forEach(file => {
            const solidityPath = path.join(contractSrcPath, file)
            const outputPath = path.join(TARGET_PATH, `${path.basename(file, '.sol')}.enums.ts`)
            generateEnumsFromSolidity(solidityPath, outputPath)
                .catch(console.error)
        })
    return entries.filter(file => fs.isDirectory(file)).map(dir => generateExtra(path.join(contractSrcPath, dir)))
}

async function main() {
    copyAbiFiles()
    generateContractTypeFiles()
    // const CONTRACT_SRC_PATH = path.resolve(__dirname, CONTRACT_PROJECT_PATH, 'contracts/')
    // await generateExtra(CONTRACT_SRC_PATH)
}

main().then(() => {
    console.log("Done.")
})
    .catch(err => {
        console.error(err)
    })
