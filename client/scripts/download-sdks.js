#!/usr/bin/env node
/**
 * Mobile SDK Auto Download Script
 * 
 * Resolves version inconsistency issues from manual SDK downloads.
 * Ensures all developers use identical SDKs by locking versions + SHA256 verification.
 * 
 * Supported Platforms:
 *   - iOS: WeChatOpenSDK, AlipaySDK
 *   - Android: WeChat SDK, Alipay SDK
 * 
 * Usage:
 *   node scripts/download-sdks.js          # Download all SDKs
 *   node scripts/download-sdks.js --ios    # Download iOS SDKs only
 *   node scripts/download-sdks.js --android # Download Android SDKs only
 * 
 * Environment Variables:
 *   WECHAT_IOS_SDK_URL    - Custom WeChat iOS SDK download URL
 *   ALIPAY_IOS_SDK_URL    - Custom Alipay iOS SDK download URL
 *   WECHAT_ANDROID_SDK_URL - Custom WeChat Android SDK download URL
 *   ALIPAY_ANDROID_SDK_URL - Custom Alipay Android SDK download URL
 *   SDK_CACHE_DIR         - Cache directory (default: ./.sdk-cache)
 */

const fs = require('fs')
const path = require('path')
const https = require('https')
const http = require('http')
const crypto = require('crypto')
const { execSync } = require('child_process')

// ==========================================
// Version Configuration - Modify here to upgrade SDK versions
// ==========================================
const SDK_CONFIG = {
    ios: {
        wechat: {
            name: 'WeChatOpenSDK',
            version: '2.0.5',
            url: process.env.WECHAT_IOS_SDK_URL || 'https://dldir1.qq.com/WechatWebDev/opensdk/OpenSDK2.0.5.zip',
            sha256: process.env.WECHAT_IOS_SHA256 || '',
            targetDir: path.join(__dirname, '../turbo-module/wechat-pay/ios'),
            frameworkName: 'WeChatOpenSDK.xcframework',
            downloadType: 'direct', // direct, cocoapods, manual
        },
        alipay: {
            name: 'AlipaySDK',
            version: '15.8.40',
            url: process.env.ALIPAY_IOS_SDK_URL || 'https://mdn.alipayobjects.com/portal_mdssth/afts/file/A*uSEbRIcKuJMAAAAAgcAAAAgAAQAAAQ/AlipaySDK-standard-15.8.40.1.zip',
            sha256: process.env.ALIPAY_IOS_SHA256 || '',
            targetDir: path.join(__dirname, '../turbo-module/alipay/ios'),
            frameworkName: 'AlipaySDK.xcframework',
            downloadType: 'direct', // direct, cocoapods, manual
            cocoapodName: 'AlipaySDK-iOS',
        },
    },
    android: {
        wechat: {
            name: 'WeChat SDK',
            version: '6.8.0',
            url: process.env.WECHAT_ANDROID_SDK_URL || '',
            sha256: process.env.WECHAT_ANDROID_SHA256 || '',
            targetDir: path.join(__dirname, '../modules/wechat-pay-module/android/libs'),
            fileName: 'libwechat-sdk-core.aar',
            downloadType: 'manual', // direct, maven, manual
            mavenDependency: 'com.tencent.mm.opensdk:wechat-sdk-android:6.8.0',
        },
        alipay: {
            name: 'Alipay SDK',
            version: '15.8.40',
            url: process.env.ALIPAY_ANDROID_SDK_URL || '',
            sha256: process.env.ALIPAY_ANDROID_SHA256 || '',
            targetDir: path.join(__dirname, '../modules/alipay-module/android/libs'),
            fileName: 'alipaysdk-15.8.40.aar',
            downloadType: 'maven', // direct, maven, manual
            mavenDependency: 'com.alipay.sdk:alipaysdk:15.8.40',
        },
    },
}

const CACHE_DIR = process.env.SDK_CACHE_DIR || path.join(__dirname, '../.sdk-cache')

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
        const protocol = url.startsWith('https') ? https : http
        const file = fs.createWriteStream(dest)
        console.log(`  Downloading: ${url}`)
        
        protocol
            .get(url, { timeout: 120000 }, (response) => {
                if (response.statusCode === 301 || response.statusCode === 302) {
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
    try {
        execSync(`unzip -o -q "${zipPath}" -d "${extractTo}"`, { stdio: 'inherit' })
    } catch {
        console.log('  System unzip not available, please manually extract or install unzip')
        throw new Error('unzip failed')
    }
}

function findFile(dir, fileName) {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name)
        if (entry.isFile() && entry.name === fileName) {
            return fullPath
        }
        if (entry.isDirectory()) {
            const found = findFile(fullPath, fileName)
            if (found) return found
        }
    }
    return null
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
// Platform Specific Functions
// ==========================================

async function downloadIOSSDK(key, config) {
    console.log(`\n📦 iOS - ${config.name} v${config.version}`)

    const frameworkPath = path.join(config.targetDir, config.frameworkName)

    if (fs.existsSync(frameworkPath)) {
        console.log(`  ✅ Already exists: ${frameworkPath}`)
        return
    }

    switch (config.downloadType) {
        case 'direct':
            await downloadDirectSDK(key, config, '.zip', (extractDir) => {
                return findFramework(extractDir, config.frameworkName)
            })
            break
        case 'cocoapods':
            console.log(`  ⚠️  Please install via CocoaPods:`)
            console.log(`     pod '${config.cocoapodName}'`)
            break
        case 'manual':
            console.log(`  ⚠️  No download URL configured for ${config.name}`)
            console.log(`  💡 Please download manually, see README.md for instructions:`)
            console.log(`     https://github.com/your-org/meta-web-three/blob/main/client/README.md`)
            console.log(`     Then place ${config.frameworkName} to: ${config.targetDir}`)
            break
    }
}

async function downloadAndroidSDK(key, config) {
    console.log(`\n📦 Android - ${config.name} v${config.version}`)

    const filePath = path.join(config.targetDir, config.fileName)

    if (fs.existsSync(filePath)) {
        console.log(`  ✅ Already exists: ${filePath}`)
        return
    }

    switch (config.downloadType) {
        case 'direct':
            await downloadDirectSDK(key, config, '.aar', (extractDir) => {
                return findFile(extractDir, config.fileName)
            })
            break
        case 'maven':
            console.log(`  ⚠️  Recommended: Use Maven dependency in build.gradle`)
            console.log(`     implementation '${config.mavenDependency}'`)
            console.log(`\n  💡 Alternatively, download manually, see README.md:`)
            console.log(`     Then place ${config.fileName} to: ${config.targetDir}`)
            break
        case 'manual':
            console.log(`  ⚠️  No download URL configured for ${config.name}`)
            console.log(`  💡 Please download manually, see README.md for instructions:`)
            console.log(`     Then place ${config.fileName} to: ${config.targetDir}`)
            break
    }
}

async function downloadDirectSDK(key, config, fileExtension, findFn) {
    if (!config.url) {
        console.log(`  ⚠️  No download URL configured`)
        return
    }

    ensureDir(CACHE_DIR)
    const cacheName = `${key}-${config.version}${fileExtension}`
    const cachePath = path.join(CACHE_DIR, cacheName)

    if (!fs.existsSync(cachePath)) {
        console.log(`  ⬇️  Downloading SDK...`)
        try {
            await downloadFile(config.url, cachePath)
        } catch (err) {
            console.error(`  ❌ Download failed: ${err.message}`)
            console.error(`  💡 Tip: Please download manually and place at: ${cachePath}`)
            throw err
        }
    } else {
        console.log(`  📋 Using cache: ${cachePath}`)
    }

    if (config.sha256) {
        const actual = sha256File(cachePath)
        if (actual !== config.sha256) {
            fs.unlinkSync(cachePath)
            throw new Error(`SHA256 verification failed! Expected: ${config.sha256}, Actual: ${actual}`)
        }
        console.log(`  🔐 SHA256 verified`)
    } else {
        console.log(`  ⚠️  SHA256 not configured, skipping verification`)
        console.log(`     Current SHA256: ${sha256File(cachePath)}`)
    }

    if (cachePath.endsWith('.zip')) {
        const extractDir = path.join(CACHE_DIR, `${key}-${config.version}-extracted`)
        console.log(`  📂 Extracting...`)
        unzip(cachePath, extractDir)

        const found = findFn(extractDir)
        if (!found) {
            throw new Error(`Could not find required file in extracted directory`)
        }

        ensureDir(config.targetDir)
        const destPath = path.join(config.targetDir, path.basename(found))
        execSync(`cp -R "${found}" "${destPath}"`)
        console.log(`  ✅ Placed at: ${destPath}`)

        execSync(`rm -rf "${extractDir}"`)
    } else {
        ensureDir(config.targetDir)
        execSync(`cp "${cachePath}" "${config.targetDir}"`)
        console.log(`  ✅ Copied to: ${config.targetDir}`)
    }
}

// ==========================================
// Main Logic
// ==========================================

async function main() {
    console.log('========================================')
    console.log('  Mobile SDK Auto Download Tool')
    console.log('========================================')

    const args = process.argv.slice(2)
    const platforms = []
    
    if (args.includes('--ios')) {
        platforms.push('ios')
    }
    if (args.includes('--android')) {
        platforms.push('android')
    }
    if (platforms.length === 0) {
        platforms.push('ios', 'android')
    }

    for (const platform of platforms) {
        console.log(`\n=== ${platform.toUpperCase()} Platform ===")
        const sdkKeys = Object.keys(SDK_CONFIG[platform])
        for (const key of sdkKeys) {
            if (platform === 'ios') {
                await downloadIOSSDK(key, SDK_CONFIG.ios[key])
            } else {
                await downloadAndroidSDK(key, SDK_CONFIG.android[key])
            }
        }
    }

    console.log('\n========================================')
    console.log('  All Done!')
    console.log('========================================')
    console.log('\nNext Steps:')
    console.log('  iOS: cd ios && pod install && cd ..')
    console.log('  Android: cd android && ./gradlew clean build')
}

main().catch((err) => {
    console.error('\n❌ Error:', err.message)
    process.exit(1)
})
