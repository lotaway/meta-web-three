import { useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Colors, Spacing, FontSizes, BorderRadius, Transitions } from './styles/constants'
import { Warehouse, Shelf } from './Warehouse3DView'

export interface WarehouseStatusProps {
  warehouse: Warehouse
  shelves: Shelf[]
  compact?: boolean
}

const STATUS_COLORS = { active: Colors.success, maintenance: Colors.warning, inactive: Colors.textMuted } as const
const UTIL_COLORS = [{ threshold: 90, color: Colors.danger }, { threshold: 70, color: Colors.warning }, { threshold: 50, color: Colors.success }, { threshold: 0, color: Colors.info }] as const

function getStatusColor(status: string): string {
  return STATUS_COLORS[status as keyof typeof STATUS_COLORS] || Colors.textSecondary
}

function getUtilizationColor(rate: number): string {
  for (const { threshold, color } of UTIL_COLORS) {
    if (rate >= threshold) return color
  }
  return Colors.info
}

interface StatCardProps {
  label: string
  value: string | number
  color?: string
}

function StatCard({ label, value, color = Colors.text }: StatCardProps) {
  return (
    <div style={{ background: Colors.backgroundSecondary, borderRadius: BorderRadius.lg, padding: Spacing.lg, textAlign: 'center' }}>
      <div style={{ color: Colors.textSecondary, fontSize: FontSizes.sm, marginBottom: Spacing.xs, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: FontSizes.title, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

function ProgressBar({ value, max = 100, color }: { value: number; max?: number; color: string }) {
  const percent = Math.min((value / max) * 100, 100)
  return (
    <div style={{ height: '10px', background: Colors.borderLight, borderRadius: BorderRadius.md, overflow: 'hidden' }}>
      <div style={{ width: `${percent}%`, height: '100%', background: color, borderRadius: BorderRadius.md, transition: Transitions.slow }} />
    </div>
  )
}

export function WarehouseStatus({ warehouse, shelves, compact = false }: WarehouseStatusProps) {
  const stats = useMemo(() => {
    const total = shelves.length
    const occupied = shelves.filter(s => s.status === 'occupied').length
    const full = shelves.filter(s => s.status === 'full').length
    const empty = shelves.filter(s => s.status === 'empty').length
    const maintenance = shelves.filter(s => s.status === 'maintenance').length
    const totalCap = shelves.reduce((sum, s) => sum + s.capacity, 0)
    const usedCap = shelves.reduce((sum, s) => sum + s.currentLoad, 0)
    return { total, occupied, full, empty, maintenance, totalCap, usedCap, rate: totalCap > 0 ? (usedCap / totalCap) * 100 : 0 }
  }, [shelves])

  const statusColor = getStatusColor(warehouse.status)
  const utilColor = getUtilizationColor(stats.rate)

  if (compact) {
    return (
      <div role="status" aria-label={`${warehouse.name} 状态`} style={{ display: 'flex', alignItems: 'center', gap: Spacing.lg, padding: `${Spacing.sm} ${Spacing.lg}`, background: Colors.background, borderRadius: BorderRadius.lg, border: `1px solid ${Colors.border}` }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
        <div style={{ color: Colors.text, fontWeight: 500, fontSize: FontSizes.lg }}>{warehouse.name}</div>
        <div style={{ color: Colors.textSecondary, fontSize: FontSizes.md }}>{stats.rate.toFixed(0)}%</div>
        <div style={{ width: '60px', height: '6px', background: Colors.borderLight, borderRadius: BorderRadius.sm, overflow: 'hidden' }}>
          <div style={{ width: `${stats.rate}%`, height: '100%', background: utilColor, borderRadius: BorderRadius.sm, transition: Transitions.normal }} />
        </div>
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <Card style={{ padding: Spacing.xl }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: Spacing.xl, paddingBottom: Spacing.lg, borderBottom: `1px solid ${Colors.border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: Spacing.sm }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: statusColor, boxShadow: `0 0 10px ${statusColor}` }} />
            <span style={{ fontWeight: 600, fontSize: FontSizes.xl }}>{warehouse.name}</span>
          </div>
          <span style={{ fontSize: FontSizes.sm, padding: `${Spacing.xs} ${Spacing.md}`, borderRadius: BorderRadius.md, background: warehouse.status === 'active' ? 'rgba(34, 197, 94, 0.2)' : warehouse.status === 'maintenance' ? 'rgba(251, 191, 36, 0.2)' : 'rgba(100, 116, 139, 0.2)', color: statusColor, fontWeight: 500, textTransform: 'uppercase' }}>{warehouse.status}</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: Spacing.lg, marginBottom: Spacing.xl }}>
          <StatCard label="面积利用率" value={`${stats.rate.toFixed(1)}%`} color={utilColor} />
          <StatCard label="库位占用" value={`${stats.occupied}/${stats.total}`} color={Colors.primary} />
        </div>
        <div style={{ marginBottom: Spacing.xl }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: Spacing.xs, fontSize: FontSizes.md }}>
            <span style={{ color: Colors.textSecondary }}>容量进度</span>
            <span style={{ color: Colors.text }}>{stats.usedCap} / {stats.totalCap}</span>
          </div>
          <ProgressBar value={stats.rate} max={100} color={utilColor} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: Spacing.md }}>
          {[{ label: '占用', value: stats.occupied, color: Colors.success }, { label: '满载', value: stats.full, color: Colors.danger }, { label: '空闲', value: stats.empty, color: Colors.textMuted }, { label: '维护', value: stats.maintenance, color: Colors.warning }].map(item => (
            <div key={item.label} style={{ background: Colors.backgroundTertiary, borderRadius: BorderRadius.md, padding: Spacing.md, textAlign: 'center' }}>
              <div style={{ color: item.color, fontWeight: 600, fontSize: '16px' }}>{item.value}</div>
              <div style={{ color: Colors.textMuted, fontSize: FontSizes.xs }}>{item.label}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: Spacing.xl, paddingTop: Spacing.lg, borderTop: `1px solid ${Colors.border}`, display: 'flex', justifyContent: 'space-between', fontSize: FontSizes.md }}>
          <div><span style={{ color: Colors.textMuted }}>总面积: </span><span style={{ color: Colors.text }}>{warehouse.totalArea} m²</span></div>
          <div><span style={{ color: Colors.textMuted }}>使用面积: </span><span style={{ color: Colors.success }}>{warehouse.usedArea} m²</span></div>
          <div><span style={{ color: Colors.textMuted }}>容量: </span><span style={{ color: Colors.primary }}>{warehouse.capacity}</span></div>
        </div>
      </Card>
    </ErrorBoundary>
  )
}