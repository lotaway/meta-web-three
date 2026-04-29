const path = require('path')
const fs = require('fs')

const projectRoot = __dirname

function getLocalModules() {
  const modules = []
  const dirs = ['modules', 'turbo-module']

  for (const dir of dirs) {
    const dirPath = path.join(projectRoot, dir)
    if (!fs.existsSync(dirPath)) continue

    const entries = fs.readdirSync(dirPath, { withFileTypes: true })
    for (const entry of entries) {
      if (!entry.isDirectory()) continue

      const modulePath = path.join(dirPath, entry.name)
      const iosPath = path.join(modulePath, 'ios')

      if (fs.existsSync(iosPath)) {
        const files = fs.readdirSync(iosPath)
        const hasPodspec = files.some(f => f.endsWith('.podspec'))

        if (hasPodspec) {
          modules.push({
            name: entry.name.replace(/-/g, '').replace(/^([a-z])/, (_, c) => c.toUpperCase()).replace(/([a-z])([A-Z])/g, '$1$2'),
            root: modulePath,
          })
        }
      }
    }
  }

  return modules
}

module.exports = {
  dependencies: getLocalModules().reduce((acc, mod) => {
    acc[mod.name] = { root: mod.root }
    return acc
  }, {}),
}