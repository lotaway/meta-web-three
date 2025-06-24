const hre = require("hardhat")
// const ethers = hre.ethers
const {join} = require("node:path")
const {createWriteStream} = require("node:fs")

export default async function({deployments, getNamedAccounts, network, solNames}) {
    // const AccountTransferContract = await hre.ethers.getContractFactory("AccountTransfer")
    // const accountTransferContract = await AccountTransferContract.deploy()
    // await accountTransferContract.deployed()
    // console.log(`AccountTransfer already deployed, address: ${accountTransferContract.address}`)
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
   const adminAddress = getNamedAccounts().admin
    const tokenContract = await hre.ethers.getContractFactory("MetaThreeCoin")
    const solName = "MetaThreeCoinFactory"
    const FactoryContract = await hre.ethers.getContractFactory(solName)
    const factoryContract = await FactoryContract.deploy({
        // from: hre.ethers.provider,
        from: adminAddress,
        args: [],
        log: true,
        contract: solName,
        proxy: {
            owner: hre.ethers.provider,
            owner: adminAddress,
            proxyContract: 'UUPS',
            execute: {
                init: {
                    methodName: 'initialize',
                    args: [],
                }
            },
            upgradeFunction: {
                methodName: "upgradeToAndCall",
                upgradeArgs: ['{implementation}', '{data}']
            }
        }
    })
    // await factoryContract.deployed()
    await factoryContract.waitForDeployment()
    // const tx = await factoryContract.deployTransaction()
    // await tx.wait(5)
    console.info(`${solName} already deployed, address: ${factoryContract.address}`)
    // await Promise.all(compiles)
    // const name = await hre.ethers.provider.getNetwork()
    // const needVerify = !!(name === "hardhat" || name === "localhost") // product verify
    const needVerify = false
    if (needVerify) {
        const verifyTX = await hre.run("verify:verify", {
            address: factoryContract.address,
            constructorArguments: []
        })
        console.info(`verifyTX: ${verifyTX}`)
    }
}