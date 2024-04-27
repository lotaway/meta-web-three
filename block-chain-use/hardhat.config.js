require("@nomiclabs/hardhat-waffle");
// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more
/** @type import('hardhat/config').HardhatUserConfig */
const accounts = process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
module.exports = {
    solidity: "0.8.20",
    paths: {
        sources: "./contracts"
    },
    networks: {
        localhost: {
            url: "http://127.0.0.1:8545"
        },
        ganache: {
            url: "http://127.0.0.1:7545",
            accounts
        },
        sepolia: {
            url: "https://eth-sepolia.alchemyapi.io/v2/liTsZpkvhffOegSsOSi-DAaOzu",
            accounts
        }
    }
};
