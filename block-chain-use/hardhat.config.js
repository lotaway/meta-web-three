require("@nomiclabs/hardhat-waffle")
require("dotenv").config()

// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more
/** @type import('hardhat/config').HardhatUserConfig */
const accounts = process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
module.exports = {
    typechain: {
        outDir: "./typechain",
        target: "ethers-v6",
    },
    solidity: {
        version: "0.8.20",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    },
    paths: {
        sources: "./contracts",
        artifacts: "./artifacts"
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
            url: `https://eth-sepolia.alchemyapi.io/v2/${process.env.NETWORK_KEY}`,
            accounts
        }
    }
};
