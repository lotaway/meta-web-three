import { useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Colors, Spacing, FontSizes, BorderRadius, Transitions } from './styles/constants'
import { Shelf, WarehouseHeatmapData } from './Warehouse3DView'

export interface ShelfHeatmapProps {
  shelves: Shelf[]
  heatmapData: WarehouseHeatmapData[]
  onCellClick?: (shelfId: string, level: number, column: number) => void
  selectedCell?: { shelfId: string; level: number; column: number } | null
  showLabels?: boolean
}

const HEATMAP_COLORS = [Colors.textMuted, Colors.heatmap.cold, Colors.heatmap.cool, Colors.heatmap.normal, Colors.heatmap.warm, Colors.heatmap.hot]
const LOAD_COLORS = [Colors.textMuted, Colors.success, Colors.warning, Colors.danger]

function getHeatmapColor(value: number, hasShelf: boolean): string {
  if (!hasShelf) return 'transparent'
  if (value === 0) return Colors.borderLight
  if (value < 20) return Colors.heatmap.cold
  if (value < 40) return Colors.heatmap.cool
  if (value < 60) return Colors.heatmap.normal
  if (value < 80) return Colors.heatmap.warm
  return Colors.heatmap.hot
}

function getLoadColor(load: number, capacity: number): string {
  const ratio = capacity > 0 ? load / capacity : 0
  if (ratio === 0) return Colors.textMuted
  if (ratio < 0.3) return Colors.success
  if (ratio < 0.7) return Colors.warning
  return Colors.danger
}

interface HeatmapCellProps {
  shelf: Shelf | null
  heatmapValue: number
  isSelected: boolean
  onClick?: () => void
}

function HeatmapCell({ shelf, heatmapValue, isSelected, onClick }: HeatmapCellProps) {
  const color = heatmapValue > 0 ? getHeatmapColor(heatmapValue, !!shelf) : (shelf ? getLoadColor(shelf.currentLoad, shelf.capacity) : Colors.background)
  return (
    <div role="gridcell" tabIndex={0} onClick={onClick} onKeyDown={(e) => e.key === 'Enter' && onClick?.()} style={{ width: '38px', height: '38px', background: color, borderRadius: BorderRadius.md, cursor: shelf ? 'pointer' : 'default', border: isSelected ? `2px solid ${Colors.primary}` : `1px solid ${Colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: FontSizes.xs, color: Colors.text, transition: Transitions.normal, opacity: shelf ? 1 : 0.3, boxShadow: heatmapValue > 60 ? `inset 0 0 8px rgba(239, 68, 68, 0.5)` : 'none' }} title={shelf ? `${shelf.code}\n负载: ${shelf.currentLoad}/${shelf.capacity}\n状态: ${shelf.status}` : '空位'}>
      {shelf && <span style={{ fontWeight: 600, fontSize: '9px', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>{shelf.currentLoad}</span>}
    </div>
  )
}

export function ShelfHeatmap({ shelves, heatmapData, onCellClick, selectedCell, showLabels = true }: ShelfHeatmapProps) {
  const gridData = useMemo(() => {
    const maxLevel = Math.max(...shelves.map(s => s.level), 0) + 1
    const maxColumn = Math.max(...shelves.map(s => s.column), 0) + 1
    const matrix: { shelf: Shelf | null; heatmapValue: number }[][] = []
    for (let level = 0; level < maxLevel; level++) {
      const row: { shelf: Shelf | null; heatmapValue: number }[] = []
      for (let column = 0; column < maxColumn; column++) {
        const shelf = shelves.find(s => s.level === level && s.column === column) || null
        const heatmap = heatmapData.find(d => d.shelfId === shelf?.id && d.level === level && d.column === column)
        row.push({ shelf, heatmapValue: heatmap?.value || 0 })
      }
      matrix.push(row)
    }
    return { matrix, maxLevel, maxColumn }
  }, [shelves, heatmapData])

  const summary = useMemo(() => ({
    total: shelves.length,
    active: shelves.filter(s => s.status === 'occupied').length,
    hot: heatmapData.filter(d => d.value >= 80).length
  }), [shelves, heatmapData])

  return (
    <ErrorBoundary>
      <Card style={{ padding: Spacing.xl }}>
        {showLabels && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: Spacing.xl, paddingBottom: Spacing.lg, borderBottom: `1px solid ${Colors.border}` }}>
            <span style={{ fontWeight: 600, fontSize: FontSizes.xl }}>库位热度图</span>
            <div style={{ display: 'flex', gap: Spacing.lg, fontSize: FontSizes.sm }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: Spacing.xs }}><span style={{ width: '12px', height: '12px', background: Colors.heatmap.cold, borderRadius: BorderRadius.sm }} />低</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: Spacing.xs }}><span style={{ width: '12px', height: '12px', background: Colors.heatmap.normal, borderRadius: BorderRadius.sm }} />中</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: Spacing.xs }}><span style={{ width: '12px', height: '12px', background: Colors.heatmap.hot, borderRadius: BorderRadius.sm }} />高</span>
            </div>
          </div>
        )}
        <div style={{ overflowX: 'auto' }}>
          <div style={{ display: 'grid', gridTemplateColumns: `30px repeat(${gridData.maxColumn}, 40px)`, gap: '2px', minWidth: 'fit-content' }}>
            <div />
            {Array.from({ length: gridData.maxColumn }).map((_, colIdx) => <div key={colIdx} style={{ textAlign: 'center', fontSize: FontSizes.xs, color: Colors.textMuted, padding: `${Spacing.xs} 0` }}>{colIdx + 1}</div>)}
            {gridData.matrix.map((row, levelIdx) => (
              <div key={levelIdx} style={{ display: 'contents' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: FontSizes.xs, color: Colors.textMuted, background: Colors.backgroundSecondary, borderRadius: BorderRadius.md }}>{levelIdx + 1}</div>
                {row.map((cell, colIdx) => (
                  <HeatmapCell key={`${levelIdx}-${colIdx}`} shelf={cell.shelf} heatmapValue={cell.heatmapValue} isSelected={!!(cell.shelf && selectedCell?.shelfId === cell.shelf.id && selectedCell?.level === levelIdx && selectedCell?.column === colIdx)} onClick={() => { if (cell.shelf) onCellClick?.(cell.shelf.id, levelIdx, colIdx) }} />
                ))}
              </div>
            ))}
          </div>
        </div>
        <div style={{ marginTop: Spacing.xl, paddingTop: Spacing.lg, borderTop: `1px solid ${Colors.border}`, display: 'flex', justifyContent: 'space-around', fontSize: FontSizes.md }}>
          <div><span style={{ color: Colors.textMuted }}>总库位: </span><span style={{ color: Colors.text }}>{summary.total}</span></div>
          <div><span style={{ color: Colors.textMuted }}>活跃: </span><span style={{ color: Colors.success }}>{summary.active}</span></div>
          <div><span style={{ color: Colors.textMuted }}>高热度: </span><span style={{ color: Colors.danger }}>{summary.hot}</span></div>
        </div>
      </Card>
    </ErrorBoundary>
  )
}