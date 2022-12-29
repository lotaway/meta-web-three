const hre = require("hardhat");

const main = async () => {
    const TransactionsContract = await hre.ethers.getContractFactory("Transactions");
    const transactionsContract = await TransactionsContract.deploy();
    await transactionsContract.deployed();
    console.log(`Transactions already deployed, address: ${transactionsContract.address}`);
};

const runMain = async () => {
    try {
        await main();
        process.exit(0);
    } catch (err) {
        console.error(err);
        process.exit(1);
    }
};

runMain();