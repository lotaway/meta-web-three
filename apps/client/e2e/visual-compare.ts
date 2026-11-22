import fs from 'fs'
import { PNG } from 'pngjs'

export interface VisualDiffResult {
  ratio: number
  changedPixels: number
  totalPixels: number
}

const IGNORE_TOP_PX = 100 // status bar is nondeterministic (clock, signal)

export function compareScreenshots(baselinePath: string, actualPath: string): VisualDiffResult {
  const baseline = PNG.sync.read(fs.readFileSync(baselinePath))
  const actual = PNG.sync.read(fs.readFileSync(actualPath))

  if (baseline.width !== actual.width || baseline.height !== actual.height) {
    throw new Error(`screenshot size mismatch: baseline ${baseline.width}x${baseline.height} vs actual ${actual.width}x${actual.height}`)
  }

  let changed = 0
  let total = 0
  const startY = Math.min(IGNORE_TOP_PX, baseline.height)
  for (let y = startY; y < baseline.height; y++) {
    for (let x = 0; x < baseline.width; x++) {
      const idx = (baseline.width * y + x) << 2
      const dr = Math.abs(baseline.data[idx] - actual.data[idx])
      const dg = Math.abs(baseline.data[idx + 1] - actual.data[idx + 1])
      const db = Math.abs(baseline.data[idx + 2] - actual.data[idx + 2])
      total++
      if (dr + dg + db > 30) changed++ // per-channel delta budget from quality config
    }
  }

  return { ratio: total === 0 ? 0 : changed / total, changedPixels: changed, totalPixels: total }
}
