const bs58 = require('bs58')
const fs = require('fs')

const secretKeyBase58 = ''  // 你的 base58 私钥字符串
const secretKey = bs58.decode(secretKeyBase58)

fs.writeFileSync('id.json', JSON.stringify(Array.from(secretKey)))
