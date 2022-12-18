const main = async () => {
    const Transactions = await hre.ethers.getContractFactory("Transactions");
    const transactions = await Transactions.deploy();
    await transactions.deployed();
    console.log(`Transactions already deployed, address: ${transactions.address}`);
};

const runMain = async () => {
    try {
        await main();
        process.exit(0);
    } catch (err) {
        console.error(error);
        process.exit(1);
    }
};