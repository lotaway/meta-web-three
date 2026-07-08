const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')

const androidDir = path.resolve(__dirname, '..', 'android')

if (process.platform === 'darwin') {
  execSync(`open -a '/Applications/Android Studio.app' "${androidDir}"`, { stdio: 'inherit' })
} else if (process.platform === 'win32') {
  const studioPaths = [
    'C:\\Program Files\\Android\\Android Studio\\bin\\studio64.exe',
    'C:\\Program Files (x86)\\Android\\Android Studio\\bin\\studio64.exe',
  ]
  const studio = studioPaths.find(p => fs.existsSync(p))
  if (!studio) {
    console.error('Android Studio not found. Open android/ folder manually.')
    process.exit(1)
  }
  execSync(`"${studio}" "${androidDir}"`, { stdio: 'inherit' })
} else {
  console.error('Unsupported platform')
  process.exit(1)
}
