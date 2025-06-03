import fs from 'fs'
import path from 'path'

const SRC_DIR = './programs'
const OUT_FILE = './generated/pda-constants.ts'

const rustFiles = []

function walk(dir) {
  const entries = fs.readdirSync(dir)
  for (const entry of entries) {
    const full = path.join(dir, entry)
    const stat = fs.statSync(full)
    if (stat.isDirectory()) walk(full)
    else if (entry.endsWith('.rs')) rustFiles.push(full)
  }
}

walk(SRC_DIR)

const results = []

for (const file of rustFiles) {
  const content = fs.readFileSync(file, 'utf8')

  // 匹配所有 struct，捕获名字和体内容
  const structRe = /pub\s+struct\s+(\w+)\s*{([^}]*)}/gs
  let structMatch
  while ((structMatch = structRe.exec(content)) !== null) {
    const structName = structMatch[1]
    const structBody = structMatch[2]

    // 在 struct 内匹配带 seeds 注解的字段，捕获注解和字段名
    // 这里匹配类似：
    // #[account(seeds = [b"xxx"], bump, signer)]
    // pub mint_authority: AccountInfo<'info>
    const fieldRe = /#\[account\s*\(([^)]*seeds\s*=\s*\[[^\]]+\][^)]*)\)\]\s*pub\s+(\w+):/g
    let fieldMatch
    while ((fieldMatch = fieldRe.exec(structBody)) !== null) {
      const attr = fieldMatch[1]
      const fieldName = fieldMatch[2]

      const seedMatch = attr.match(/seeds\s*=\s*\[([^\]]+)\]/)
      if (!seedMatch) continue

      const rawSeeds = seedMatch[1]
      const seeds = rawSeeds.split(',').map(s => s.trim()).filter(Boolean)

      seeds.forEach((seed, idx) => {
        let constName = `${structName}_${fieldName}_Seed${idx}`
        let value = ''

        if (/^b?["'].*["']$/.test(seed)) {
          value = `Buffer.from(${seed.replace(/^b?/, '')})`
        } else if (/^\[.*\]$/.test(seed)) {
          value = `Buffer.from(${seed})`
        } else {
          value = seed
        }

        results.push(`export const ${constName} = ${value}`)
      })
    }
  }
}

fs.writeFileSync(OUT_FILE, results.join('\n') + '\n')
console.log(`✅ Generated: ${OUT_FILE}`)
