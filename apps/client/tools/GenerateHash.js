const fs = require('fs')
const crypto = require('crypto')

function getFileHash(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);

    stream.on('data', (chunk) => {
      hash.update(chunk)
    })

    stream.on('end', () => {
      resolve(hash.digest('hex'))
    })

    stream.on('error', (err) => {
      reject(err)
    })
  })
}

getFileHash('path/to/large/file')
  .then((hashValue) => console.log('File Hash:', hashValue))
  .catch((err) => console.error('Error:', err))