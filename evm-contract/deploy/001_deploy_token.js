const hre = require("hardhat")

module.exports = async function ({deployments, getNamedAccounts}) {
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
    const {deploy} = deployments
    const {admin: adminAddress} = await getNamedAccounts()
    const solName = "MetaThreeCoinFactory"

    const factoryDeployment = await deploy(solName, {
        from: adminAddress,
        log: true,
        proxy: {
            owner: adminAddress,
            proxyContract: "UUPS",
            execute: {
                init: {
                    methodName: "initialize",
                    args: []
                }
            }
        }
    })
    // const tx = await factoryContract.deployTransaction()
    // await tx.wait(5)
    console.info(`${solName} deployed, address: ${factoryDeployment.address}`)
    // await Promise.all(compiles)
    // const name = await hre.ethers.provider.getNetwork()
    // const needVerify = !!(name === "hardhat" || name === "localhost") // product verify
    const needVerify = false
    if (needVerify) {
        const verifyTX = await hre.run("verify:verify", {
            address: factoryDeployment.address,
            constructorArguments: []
        })
        console.info(`verifyTX: ${verifyTX}`)
    }
}
