#!/usr/bin/env node
/**
 * iOS SDK Auto Download Script
 *
 * Resolves version inconsistency issues from manual SDK downloads.
 * Ensures all developers use identical SDKs by locking versions + SHA256 verification.
 *
 * Usage:
 *   node scripts/download-ios-sdks.js
 *
 * Environment Variables:
 *   WECHAT_SDK_URL    - Custom WeChat SDK download URL
 *   ALIPAY_SDK_URL    - Custom Alipay SDK download URL
 *   IOS_SDK_CACHE_DIR - Cache directory (default: ./.ios-sdk-cache)
 */

const fs = require('fs')
const path = require('path')
const https = require('https')
const crypto = require('crypto')
const { execSync } = require('child_process')

// ==========================================
// Version Configuration - Modify here to upgrade SDK versions
// ==========================================
const SDK_CONFIG = {
    wechat: {
        name: 'WeChatOpenSDK',
        version: '2.0.5',
        // Official download page: https://developers.weixin.qq.com/doc/oplatform/Mobile_App/Downloads/iOS_Resource.html
        // Note: WeChat official does not provide direct download links. Replace with internal CDN or reliable mirror.
        url: process.env.WECHAT_SDK_URL || 'https://dldir1.qq.com/WechatWebDev/opensdk/XCFramework/OpenSDK2.0.5.zip',
        sha256: process.env.WECHAT_SDK_SHA256 || '', // Fill in actual SHA256
        targetDir: path.join(__dirname, '../turbo-module/wechat-pay/ios'),
        frameworkName: 'WeChatOpenSDK.xcframework',
    },
    alipay: {
        name: 'AlipaySDKCore',
        version: '15.8.16',
        // Official download page: https://opendocs.alipay.com/open/204/105295/
        // Note: Alipay official does not provide direct download links. Replace with internal CDN or reliable mirror.
        url: process.env.ALIPAY_SDK_URL || 'https://github.com/alipay/alipay-sdk-ios/releases/download/15.8.16/AlipaySDK-iOS-15.8.16.zip',
        sha256: process.env.ALIPAY_SDK_SHA256 || '', // Fill in actual SHA256
        targetDir: path.join(__dirname, '../turbo-module/alipay/ios'),
        frameworkName: 'AlipaySDKCore.xcframework',
    },
}

const CACHE_DIR = process.env.IOS_SDK_CACHE_DIR || path.join(__dirname, '../.ios-sdk-cache')

// ==========================================
// Utility Functions
// ==========================================

function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
    }
}

function sha256File(filePath) {
    const hash = crypto.createHash('sha256')
    hash.update(fs.readFileSync(filePath))
    return hash.digest('hex')
}

function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest)
        console.log(`  Downloading: ${url}`)
        https
            .get(url, { timeout: 120000 }, (response) => {
                if (response.statusCode === 301 || response.statusCode === 302) {
                    // Follow redirects
                    file.close()
                    fs.unlinkSync(dest)
                    return downloadFile(response.headers.location, dest).then(resolve).catch(reject)
                }
                if (response.statusCode !== 200) {
                    file.close()
                    fs.unlinkSync(dest)
                    return reject(new Error(`HTTP ${response.statusCode}`))
                }
                response.pipe(file)
                file.on('finish', () => {
                    file.close()
                    resolve()
                })
            })
            .on('error', (err) => {
                fs.unlinkSync(dest)
                reject(err)
            })
    })
}

function unzip(zipPath, extractTo) {
    ensureDir(extractTo)
    // Prefer system unzip, fallback to node
    try {
        execSync(`unzip -o -q "${zipPath}" -d "${extractTo}"`, { stdio: 'inherit' })
    } catch {
        console.log('  System unzip not available, please manually extract or install unzip')
        throw new Error('unzip failed')
    }
}

function findFramework(dir, frameworkName) {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name)
        if (entry.isDirectory() && entry.name === frameworkName) {
            return fullPath
        }
        if (entry.isDirectory()) {
            const found = findFramework(fullPath, frameworkName)
            if (found) return found
        }
    }
    return null
}

// ==========================================
// Main Logic
// ==========================================

async function downloadSDK(key, config) {
    console.log(`\n📦 ${config.name} v${config.version}`)

    const frameworkPath = path.join(config.targetDir, config.frameworkName)

    // 1. Check if already exists
    if (fs.existsSync(frameworkPath)) {
        console.log(`  ✅ Already exists: ${frameworkPath}`)
        return
    }

    // 2. Prepare cache
    ensureDir(CACHE_DIR)
    const zipName = `${key}-${config.version}.zip`
    const cachePath = path.join(CACHE_DIR, zipName)

    // 3. Download (if cache does not exist)
    if (!fs.existsSync(cachePath)) {
        console.log(`  ⬇️  Downloading SDK...`)
        try {
            await downloadFile(config.url, cachePath)
        } catch (err) {
            console.error(`  ❌ Download failed: ${err.message}`)
            console.error(`\n💡 Tip: WeChat/Alipay official do not provide direct download links.`)
            console.error(`   Please manually download SDK to: ${cachePath}`)
            console.error(`   Or set env var ${key.toUpperCase()}_SDK_URL to internal CDN.`)
            throw err
        }
    } else {
        console.log(`  📋 Using cache: ${cachePath}`)
    }

    // 4. Verify SHA256 (if configured)
    if (config.sha256) {
        const actual = sha256File(cachePath)
        if (actual !== config.sha256) {
            fs.unlinkSync(cachePath)
            throw new Error(`SHA256 verification failed! Expected: ${config.sha256}, Actual: ${actual}`)
        }
        console.log(`  🔐 SHA256 verified`)
    } else {
        console.log(`  ⚠️  SHA256 not configured, skipping verification`)
        console.log(`     Current file SHA256: ${sha256File(cachePath)}`)
        console.log(`     Suggestion: Write the above value to SDK_CONFIG.${key}.sha256 to enable verification`)
    }

    // 5. Extract
    const extractDir = path.join(CACHE_DIR, `${key}-${config.version}-extracted`)
    console.log(`  📂 Extracting to: ${extractDir}`)
    unzip(cachePath, extractDir)

    // 6. Find .xcframework
    const foundFramework = findFramework(extractDir, config.frameworkName)
    if (!foundFramework) {
        throw new Error(`Could not find ${config.frameworkName} in extracted directory`)
    }

    // 7. Move to target directory
    ensureDir(config.targetDir)
    const destFramework = path.join(config.targetDir, config.frameworkName)
    // Use cp -R to preserve symlinks (xcframework contains symlinks internally)
    execSync(`cp -R "${foundFramework}" "${destFramework}"`)
    console.log(`  ✅ Placed at: ${destFramework}`)

    // 8. Clean up temporary extracted directory
    execSync(`rm -rf "${extractDir}"`)
}

async function main() {
    console.log('========================================')
    console.log('  iOS SDK Auto Download Tool')
    console.log('========================================')

    const keys = Object.keys(SDK_CONFIG)
    for (const key of keys) {
        await downloadSDK(key, SDK_CONFIG[key])
    }

    console.log('\n========================================')
    console.log('  All Done!')
    console.log('========================================')
    console.log('\nNext:')
    console.log('  cd ios && pod install && cd ..')
}

main().catch((err) => {
    console.error('\n❌ Error:', err.message)
    process.exit(1)
});

