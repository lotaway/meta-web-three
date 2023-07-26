const {expect} = require("chai")

describe("Transaction contract", function () {
    it("Should return the new greeting once it's changed", async function () {
        const AccountTransfer = await ethers.getContractFactory("AccountTransfer")
        const accountTransfer = await AccountTransfer.deploy()

        await accountTransfer.deployed()
        expect(await accountTransfer.getRecordCount()).to.equal(0)

        accountTransfer.addRecord("test", 1, "Hello test!", "test,dev,local")
        expect(await accountTransfer.getRecord().length).to.equal(0)
    })
})