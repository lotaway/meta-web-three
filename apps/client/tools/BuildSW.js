const {execSync} = require('child_process')

console.log("Building Service Worker...", process.env.NODE_ENV)
const isProduction = process.env.NODE_ENV === 'production'

execSync(`tsc --project ${isProduction ? "tsconfig.sw.json" : "tsconfig.sw.dev.json"}`, {stdio: 'inherit'})

if (isProduction) {
    execSync('terser public/sw.js --compress --mangle --output public/sw.js', {stdio: 'inherit'})
}
console.log("Service Worker Built")