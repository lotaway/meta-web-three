const {expect} = require("chai")
const { ethers } = require("hardhat")

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

describe("Test token", function () {
    it("Should be able to get decimals", async function() {
        const TokenContarct = await ethers.getContractFactory("MetaThreeCoin")
        const tokenContract = await TokenContarct.deploy()
        await tokenContract.deployed()
        const value = await tokenContract.decimals()
        expect(value).to.equal(4)
    })
})