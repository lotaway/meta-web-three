// Rewrite a generated `apis/index.ts` barrel so it does not re-export
// ambiguous names.
//
// The gateway aggregates the OpenAPI documents of many services. Because
// `useSingleRequestParameter=true`, `openapi-generator` derives request
// parameter interfaces from the operation name (e.g. `CreateRequest`,
// `GetByIdRequest`, `ListRequest`). Different services frequently reuse the
// same operation names, so the same interface name ends up exported by several
// `XxxApi.ts` modules. The generated barrel uses `export * from './XxxApi'`,
// which makes every one of those names ambiguous and fails the build with
// TS2308.
//
// This post-processing step parses the exported names of every API module and
// rewrites the barrel with explicit re-exports, dropping only the names that
// appear in more than one module. Those names are internal parameter types and
// are never imported by application code, so dropping them is safe.
//
// Usage:
//   node apps/tools/fixApiIndexBarrel.js [outputDir]
// or programmatically:
//   const { fixApisIndexBarrel } = require('./fixApiIndexBarrel')
//   await fixApisIndexBarrel({ outputDir: 'src/generated/api' })

const fs = require("node:fs")
const path = require("node:path")

// Matches `export interface X`, `export type X`, `export class X`,
// `export const X`, `export enum X`, `export function X`, etc.
const EXPORT_PATTERN =
  /^export\s+(?:(type)\s+)?(?:interface|class|enum|const|function|let|var)\s+([A-Za-z0-9_$]+)/gm

/**
 * @param {object} [options]
 * @param {string} [options.outputDir] Directory (relative to cwd or absolute)
 *   that contains the generated `apis/` folder. Defaults to `src/generated/api`.
 * @param {string} [options.cwd] Project root. Defaults to `process.cwd()`.
 * @returns {Promise<{ indexPath: string, modules: number, dropped: number } | null>}
 */
async function fixApisIndexBarrel(options = {}) {
  const projectRoot = options.cwd || process.cwd()
  const outputDir = options.outputDir || "src/generated/api"
  const apisDir = path.resolve(projectRoot, outputDir, "apis")
  const modelsDir = path.resolve(projectRoot, outputDir, "models")
  const indexPath = path.join(apisDir, "index.ts")

  if (!fs.existsSync(indexPath)) {
    return null
  }

  const files = (await fs.promises.readdir(apisDir))
    .filter((file) => file.endsWith(".ts") && file !== "index.ts")
    .sort()

  // Collect the names exported by each API module. A name may be declared more
  // than once in the same file (e.g. an enum `const` merged with its `type`),
  // so deduplicate per file before counting across files.
  const collect = async (dir) => {
    if (!fs.existsSync(dir)) {
      return []
    }
    const dirFiles = (await fs.promises.readdir(dir)).filter(
      (file) => file.endsWith(".ts") && file !== "index.ts"
    )
    const result = []
    for (const file of dirFiles) {
      const source = await fs.promises.readFile(path.join(dir, file), "utf-8")
      const values = new Set()
      const types = new Set()
      let match
      EXPORT_PATTERN.lastIndex = 0
      while ((match = EXPORT_PATTERN.exec(source))) {
        const name = match[2]
        const typeOnly =
          match[1] === "type" || /^\s*export\s+interface\b/.test(match[0])
        if (typeOnly) {
          types.add(name)
        } else {
          values.add(name)
        }
      }
      result.push({ file, values, types })
    }
    return result
  }

  const apiModules = await collect(apisDir)
  const modelModules = await collect(modelsDir)

  // A name is ambiguous when it is exported by more than one API module, or by
  // an API module *and* a model module (`api/index.ts` re-exports both barrels,
  // so such a name would also be ambiguous there).
  const nameCount = new Map()
  for (const { values, types } of apiModules) {
    for (const name of new Set([...values, ...types])) {
      nameCount.set(name, (nameCount.get(name) || 0) + 1)
    }
  }
  for (const { values, types } of modelModules) {
    for (const name of new Set([...values, ...types])) {
      nameCount.set(name, (nameCount.get(name) || 0) + 1)
    }
  }

  const lines = ["/* tslint:disable */", "/* eslint-disable */"]
  let dropped = 0

  for (const { file, values: allValues, types: allTypes } of apiModules) {
    const base = file.replace(/\.ts$/, "")
    const values = [...allValues].filter((name) => nameCount.get(name) === 1)
    const types = [...allTypes].filter(
      (name) => nameCount.get(name) === 1 && !allValues.has(name)
    )
    dropped += allValues.size + allTypes.size - values.length - types.length

    if (values.length > 0) {
      lines.push(`export { ${values.join(", ")} } from './${base}';`)
    }
    if (types.length > 0) {
      lines.push(`export type { ${types.join(", ")} } from './${base}';`)
    }
  }

  await fs.promises.writeFile(indexPath, lines.join("\n") + "\n")

  return { indexPath, modules: apiModules.length, dropped }
}

module.exports = { fixApisIndexBarrel }

// Allow running this file directly: `node apps/tools/fixApiIndexBarrel.js [outputDir]`
if (require.main === module) {
  const outputDir = process.argv[2] || "src/generated/api"
  fixApisIndexBarrel({ outputDir })
    .then((result) => {
      if (!result) {
        console.log(`No apis/index.ts found under ${outputDir}`)
        return
      }
      console.log(
        `Rewrote ${result.indexPath} (${result.modules} modules, skipped ${result.dropped} ambiguous re-exports).`
      )
    })
    .catch((error) => {
      console.error("fixApisIndexBarrel error:", error)
      process.exit(1)
    })
}
