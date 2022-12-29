require("@nomiclabs/hardhat-waffle");
// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: "0.8.0",
    networks: {
        sepolia: {
            url: "https://eth-sepolia.alchemyapi.io/v2/liTsZpkvhffOegSsOSi-DAaOzu",
            accounts: ["83ba30650b66d73633c7b6cc6da8ebb01d5698dbae2df05afbc554a4f7926290"]
        }
    }
};