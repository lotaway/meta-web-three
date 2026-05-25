import { useMemo } from 'react'
import { Warehouse, Shelf } from './Warehouse3DView'

export interface WarehouseStatusProps {
  warehouse: Warehouse
  shelves: Shelf[]
  compact?: boolean
}

export function WarehouseStatus({ warehouse, shelves, compact = false }: WarehouseStatusProps) {
  // Calculate statistics
  const stats = useMemo(() => {
    const totalShelves = shelves.length
    const occupiedShelves = shelves.filter(s => s.status === 'occupied').length
    const fullShelves = shelves.filter(s => s.status === 'full').length
    const emptyShelves = shelves.filter(s => s.status === 'empty').length
    const maintenanceShelves = shelves.filter(s => s.status === 'maintenance').length
    
    const totalCapacity = shelves.reduce((sum, s) => sum + s.capacity, 0)
    const usedCapacity = shelves.reduce((sum, s) => sum + s.currentLoad, 0)
    
    return {
      totalShelves,
      occupiedShelves,
      fullShelves,
      emptyShelves,
      maintenanceShelves,
      totalCapacity,
      usedCapacity,
      utilizationRate: totalCapacity > 0 ? (usedCapacity / totalCapacity) * 100 : 0
    }
  }, [shelves])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#22c55e'
      case 'maintenance': return '#fbbf24'
      case 'inactive': return '#64748b'
      default: return '#94a3b8'
    }
  }

  const getUtilizationColor = (rate: number) => {
    if (rate >= 90) return '#ef4444'
    if (rate >= 70) return '#fbbf24'
    if (rate >= 50) return '#22c55e'
    return '#3b82f6'
  }

  if (compact) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 12px',
        background: 'rgba(15, 23, 42, 0.8)',
        borderRadius: '8px',
        border: '1px solid #334155'
      }}>
        <div style={{ 
          width: '10px', 
          height: '10px', 
          borderRadius: '50%', 
          background: getStatusColor(warehouse.status),
          boxShadow: `0 0 8px ${getStatusColor(warehouse.status)}`
        }} />
        <div style={{ color: '#f1f5f9', fontWeight: 500, fontSize: '13px' }}>
          {warehouse.name}
        </div>
        <div style={{ color: '#94a3b8', fontSize: '12px' }}>
          {stats.utilizationRate.toFixed(0)}%
        </div>
        <div style={{ 
          width: '60px', 
          height: '6px', 
          background: '#1e293b', 
          borderRadius: '3px',
          overflow: 'hidden'
        }}>
          <div style={{ 
            width: `${stats.utilizationRate}%`, 
            height: '100%', 
            background: getUtilizationColor(stats.utilizationRate),
            borderRadius: '3px',
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>
    )
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
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '16px',
        paddingBottom: '12px',
        borderBottom: '1px solid #334155'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: getStatusColor(warehouse.status),
            boxShadow: `0 0 10px ${getStatusColor(warehouse.status)}`
          }} />
          <span style={{ fontWeight: 600, fontSize: '15px' }}>{warehouse.name}</span>
        </div>
        <span style={{ 
          fontSize: '11px', 
          padding: '4px 8px', 
          borderRadius: '4px',
          background: warehouse.status === 'active' ? 'rgba(34, 197, 94, 0.2)' : 
                      warehouse.status === 'maintenance' ? 'rgba(251, 191, 36, 0.2)' : 
                      'rgba(100, 116, 139, 0.2)',
          color: getStatusColor(warehouse.status),
          fontWeight: 500,
          textTransform: 'uppercase'
        }}>
          {warehouse.status}
        </span>
      </div>

      {/* Main metrics */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(2, 1fr)', 
        gap: '12px',
        marginBottom: '16px'
      }}>
        <div style={{
          background: 'rgba(30, 41, 59, 0.8)',
          borderRadius: '8px',
          padding: '12px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '4px', textTransform: 'uppercase' }}>
            面积利用率
          </div>
          <div style={{ 
            fontSize: '28px', 
            fontWeight: 700,
            color: getUtilizationColor(stats.utilizationRate)
          }}>
            {stats.utilizationRate.toFixed(1)}%
          </div>
        </div>
        
        <div style={{
          background: 'rgba(30, 41, 59, 0.8)',
          borderRadius: '8px',
          padding: '12px',
          textAlign: 'center'
        }}>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '4px', textTransform: 'uppercase' }}>
            库位占用
          </div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#38bdf8' }}>
            {stats.occupiedShelves}/{stats.totalShelves}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          marginBottom: '6px',
          fontSize: '12px'
        }}>
          <span style={{ color: '#94a3b8' }}>容量进度</span>
          <span style={{ color: '#f1f5f9' }}>
            {stats.usedCapacity} / {stats.totalCapacity}
          </span>
        </div>
        <div style={{ 
          height: '10px', 
          background: '#1e293b', 
          borderRadius: '5px',
          overflow: 'hidden'
        }}>
          <div style={{ 
            width: `${Math.min(stats.utilizationRate, 100)}%`, 
            height: '100%', 
            background: `linear-gradient(90deg, ${getUtilizationColor(stats.utilizationRate)}, ${getUtilizationColor(stats.utilizationRate)}dd)`,
            borderRadius: '5px',
            transition: 'width 0.5s ease'
          }} />
        </div>
      </div>

      {/* Shelf status breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
        {[
          { label: '占用', value: stats.occupiedShelves, color: '#22c55e' },
          { label: '满载', value: stats.fullShelves, color: '#ef4444' },
          { label: '空闲', value: stats.emptyShelves, color: '#64748b' },
          { label: '维护', value: stats.maintenanceShelves, color: '#fbbf24' }
        ].map(item => (
          <div key={item.label} style={{
            background: 'rgba(30, 41, 59, 0.6)',
            borderRadius: '6px',
            padding: '8px',
            textAlign: 'center'
          }}>
            <div style={{ color: item.color, fontWeight: 600, fontSize: '16px' }}>
              {item.value}
            </div>
            <div style={{ color: '#64748b', fontSize: '10px' }}>
              {item.label}
            </div>
          </div>
        ))}
      </div>

      {/* Area details */}
      <div style={{ 
        marginTop: '16px', 
        paddingTop: '12px', 
        borderTop: '1px solid #334155',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '12px'
      }}>
        <div>
          <span style={{ color: '#64748b' }}>总面积: </span>
          <span style={{ color: '#f1f5f9' }}>{warehouse.totalArea} m²</span>
        </div>
        <div>
          <span style={{ color: '#64748b' }}>使用面积: </span>
          <span style={{ color: '#22c55e' }}>{warehouse.usedArea} m²</span>
        </div>
        <div>
          <span style={{ color: '#64748b' }}>容量: </span>
          <span style={{ color: '#38bdf8' }}>{warehouse.capacity}</span>
        </div>
      </div>
    </div>
  )
}

export type { WarehouseStatusProps }