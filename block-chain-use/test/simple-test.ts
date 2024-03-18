const mocha = require("mocha")
const chai = require("chai")
const hardhat = require("hardhat")

// describe("Transaction contract", function () {
//     it("Should return the new greeting once it's changed", async function () {
//         const AccountTransfer = await ethers.getContractFactory("AccountTransfer")
//         const accountTransfer = await AccountTransfer.deploy()

//         await accountTransfer.deployed()
//         expect(await accountTransfer.getRecordCount()).to.equal(0)

//         accountTransfer.addRecord("test", 1, "Hello test!", "test,dev,local")
//         expect(await accountTransfer.getRecord().length).to.equal(0)
//     })
// })

mocha.describe("Test token", function () {

    mocha.beforeEach(async () => {
        const TokenContarct = await hardhat.ethers.getContractFactory("MetaThreeCoin")
        const tokenContract = await TokenContarct.deploy()
        await tokenContract.deployed()
        global.tokenContract = tokenContract
    })

    mocha.it("Should be able to get decimals", async function () {
        const value = await global.tokenContract.decimals()
        chai.expect(value).to.equal(4)
    })

    mocha.it("Should be able to get balance", async function () {
        const balance = await global.tokenContract.balanceOf("0x391fb54Eb98b21E3C777649Cdf3500AA05eD4715")
        chai.expect(balance).to.equal(0)
    })
})