require("@nomiclabs/hardhat-waffle");
// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: "0.8.20",
    paths: {
        sources: "./contracts"
    },
    networks: {
        ganache: {
            url: "http://127.0.0.1:7545",
            accounts: ["0x14a6c69405e6bc4c7a061a114681939e7d06a60ac9ca011761ef236d06447e90"]
        },
        sepolia: {
            url: "https://eth-sepolia.alchemyapi.io/v2/liTsZpkvhffOegSsOSi-DAaOzu",
            accounts: ["83ba30650b66d73633c7b6cc6da8ebb01d5698dbae2df05afbc554a4f7926290"]
        },
        bitsatTest: {
            // url: "ws://172.21.255.113:8546",
            url: "http://47.76.112.207:8545",
            // accounts: [`0x${process.env.PRIVATE_KEY}`],
            accounts: [`0x14a6c69405e6bc4c7a061a114681939e7d06a60ac9ca011761ef236d06447e90`],
            chainId: 88878
        }
    }
};
