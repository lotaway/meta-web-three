const hre = require("hardhat")
// const ethers = hre.ethers

const ethers = require("ethers")
const {join} = require("node:path")
const {createWriteStream} = require("node:fs")

async function main(options) {
    // const AccountTransferContract = await hre.ethers.getContractFactory("AccountTransfer")
    // const accountTransferContract = await AccountTransferContract.deploy()
    // await accountTransferContract.deployed()
    // console.log(`AccountTransfer already deployed, address: ${accountTransferContract.address}`)
    const solNames = [
        // "AccountTransfer",
        // "AccountTemplate",
        // "EncryptedToken",
        "MetaThreeCoin",
    ]
    // const addresses = []
    /*let compiles = solNames.map(solName => {
        return hre.ethers.getContractFactory(solName)
            .then(Contract => Contract.deploy())
            .then(contract => {
                const output = `${solName} already deployed, address: ${contract.address}`
                console.log(output)
                // writeStream.write(output + "\n")
                return contract.deployed()
            })
    })*/
    while (solNames.length) {
        const solName = solNames.shift()
        const Contract = await hre.ethers.getContractFactory(solName)
        const contract = await Contract.deploy()
        options?.output(`${solName} already deployed, address: ${contract.address}`)
    }
    // await Promise.all(compiles)
}

function runMain() {
    const logWriteStream = () => createWriteStream(join(__dirname, "../artifacts/build-info/deploy-output.txt"), {encoding: "utf-8", flags: "r+"})
    return main({
        output(message) {
            console.log(message)
            logWriteStream().write(message + "\n")
        }})
        .then(() => {
            logWriteStream().end("success")
            process.exit(0)
        })
        .catch(err => {
            const message = "error: " + JSON.stringify(err)
            console.error(message)
            logWriteStream().end(message)
            process.exit(1)
        })
}

void runMain()
