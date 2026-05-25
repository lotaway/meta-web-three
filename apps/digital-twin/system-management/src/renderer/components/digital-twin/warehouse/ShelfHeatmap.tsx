import { useMemo } from 'react'
import { Shelf, WarehouseHeatmapData } from './Warehouse3DView'

export interface ShelfHeatmapProps {
  shelves: Shelf[]
  heatmapData: WarehouseHeatmapData[]
  onCellClick?: (shelfId: string, level: number, column: number) => void
  selectedCell?: { shelfId: string; level: number; column: number } | null
  showLabels?: boolean
}

export function ShelfHeatmap({ 
  shelves, 
  heatmapData, 
  onCellClick,
  selectedCell,
  showLabels = true
}: ShelfHeatmapProps) {
  // Group shelves by rows
  const gridData = useMemo(() => {
    const maxLevel = Math.max(...shelves.map(s => s.level), 0) + 1
    const maxColumn = Math.max(...shelves.map(s => s.column), 0) + 1
    
    // Create grid matrix
    const matrix: { shelf: Shelf | null; heatmapValue: number }[][] = []
    
    for (let level = 0; level < maxLevel; level++) {
      const row: { shelf: Shelf | null; heatmapValue: number }[] = []
      for (let column = 0; column < maxColumn; column++) {
        const shelf = shelves.find(s => s.level === level && s.column === column) || null
        const heatmap = heatmapData.find(d => 
          d.shelfId === shelf?.id && d.level === level && d.column === column
        )
        row.push({
          shelf,
          heatmapValue: heatmap?.value || 0
        })
      }
      matrix.push(row)
    }
    
    return { matrix, maxLevel, maxColumn }
  }, [shelves, heatmapData])

  const getHeatmapColor = (value: number, hasShelf: boolean) => {
    if (!hasShelf) return 'transparent'
    if (value === 0) return '#1e293b'
    
    // Color gradient: blue -> cyan -> green -> yellow -> red
    if (value < 20) return '#1d4ed8'
    if (value < 40) return '#0ea5e9'
    if (value < 60) return '#22c55e'
    if (value < 80) return '#eab308'
    return '#ef4444'
  }

  const getLoadColor = (load: number, capacity: number) => {
    const ratio = capacity > 0 ? load / capacity : 0
    if (ratio === 0) return '#64748b'
    if (ratio < 0.3) return '#22c55e'
    if (ratio < 0.7) return '#eab308'
    return '#ef4444'
  }

  const isSelected = (shelfId: string, level: number, column: number) => {
    return selectedCell?.shelfId === shelfId && 
           selectedCell?.level === level && 
           selectedCell?.column === column
  }

  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.9)',
      borderRadius: '12px',
      border: '1px solid #334155',
      padding: '16px',
      color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Header */}
      {showLabels && (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          marginBottom: '16px',
          paddingBottom: '12px',
          borderBottom: '1px solid #334155'
        }}>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>库位热度图</span>
          <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ width: '12px', height: '12px', background: '#1d4ed8', borderRadius: '2px' }} />
              低
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '2px' }} />
              中
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ width: '12px', height: '12px', background: '#ef4444', borderRadius: '2px' }} />
              高
            </span>
          </div>
        </div>
      )}

      {/* Heatmap grid */}
      <div style={{ overflowX: 'auto' }}>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: `30px repeat(${gridData.maxColumn}, 40px)`,
          gap: '2px',
          minWidth: 'fit-content'
        }}>
          {/* Column headers */}
          <div />
          {Array.from({ length: gridData.maxColumn }).map((_, colIdx) => (
            <div key={colIdx} style={{
              textAlign: 'center',
              fontSize: '10px',
              color: '#64748b',
              padding: '4px 0'
            }}>
              {colIdx + 1}
            </div>
          ))}

          {/* Grid rows */}
          {gridData.matrix.map((row, levelIdx) => (
            <div key={levelIdx} style={{ display: 'contents' }}>
              {/* Row header */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                color: '#64748b',
                background: 'rgba(30, 41, 59, 0.8)',
                borderRadius: '4px'
              }}>
                {levelIdx + 1}
              </div>
              
              {/* Cells */}
              {row.map((cell, colIdx) => (
                <div
                  key={`${levelIdx}-${colIdx}`}
                  onClick={() => {
                    if (cell.shelf) {
                      onCellClick?.(cell.shelf.id, levelIdx, colIdx)
                    }
                  }}
                  style={{
                    width: '38px',
                    height: '38px',
                    background: cell.heatmapValue > 0 
                      ? getHeatmapColor(cell.heatmapValue, !!cell.shelf)
                      : (cell.shelf ? getLoadColor(cell.shelf.currentLoad, cell.shelf.capacity) : '#0f172a'),
                    borderRadius: '4px',
                    cursor: cell.shelf ? 'pointer' : 'default',
                    border: isSelected(cell.shelf?.id || '', levelIdx, colIdx) 
                      ? '2px solid #38bdf8' 
                      : '1px solid #334155',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    color: '#f1f5f9',
                    transition: 'all 0.2s ease',
                    opacity: cell.shelf ? 1 : 0.3,
                    boxShadow: cell.heatmapValue > 60 ? `inset 0 0 8px rgba(239, 68, 68, 0.5)` : 'none'
                  }}
                  title={cell.shelf 
                    ? `${cell.shelf.code}\n负载: ${cell.shelf.currentLoad}/${cell.shelf.capacity}\n状态: ${cell.shelf.status}` 
                    : '空位'}
                >
                  {cell.shelf && (
                    <span style={{ 
                      fontWeight: 600, 
                      fontSize: '9px',
                      textShadow: '0 1px 2px rgba(0,0,0,0.5)'
                    }}>
                      {cell.shelf.currentLoad}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div style={{
        marginTop: '16px',
        paddingTop: '12px',
        borderTop: '1px solid #334155',
        display: 'flex',
        justifyContent: 'space-around',
        fontSize: '12px'
      }}>
        <div>
          <span style={{ color: '#64748b' }}>总库位: </span>
          <span style={{ color: '#f1f5f9' }}>{shelves.length}</span>
        </div>
        <div>
          <span style={{ color: '#64748b' }}>活跃: </span>
          <span style={{ color: '#22c55e' }}>
            {shelves.filter(s => s.status === 'occupied').length}
          </span>
        </div>
        <div>
          <span style={{ color: '#64748b' }}>高热度: </span>
          <span style={{ color: '#ef4444' }}>
            {heatmapData.filter(d => d.value >= 80).length}
          </span>
        </div>
      </div>
    </div>
  )
}

export type { ShelfHeatmapProps }