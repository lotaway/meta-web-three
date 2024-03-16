require("@nomiclabs/hardhat-waffle");
// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: "^0.8.20",
    paths: {
        sources: "./contracts"
    },
    networks: {
        ganache: {
            url: "http://127.0.0.1:7545",
            accounts: ["0x1aef2fd848b0c53a0407b397c92e5144e79547a9f9f67c43f1182d656b2b78ea"]
        },
        sepolia: {
            url: "https://eth-sepolia.alchemyapi.io/v2/liTsZpkvhffOegSsOSi-DAaOzu",
            accounts: ["83ba30650b66d73633c7b6cc6da8ebb01d5698dbae2df05afbc554a4f7926290"]
        }
    }
};
