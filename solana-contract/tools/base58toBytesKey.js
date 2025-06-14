const bs58 = require('bs58')
const fs = require('fs')

const secretKeyBase58 = '4EgqssMAf1gLcrKvn3btqQZhcgW2BH6vVYsZv9yaQz6xj4fTyT17FksxL7kBjLBfd3vmcJDHuhxryPNNmD78u8EK'  // 你的 base58 私钥字符串
const secretKey = bs58.decode(secretKeyBase58)

fs.writeFileSync('id.json', JSON.stringify(Array.from(secretKey)))
