import { useState, useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Badge } from './components/Badge'
import { Button } from './components/Button'
import { Colors, Spacing, FontSizes, BorderRadius, Transitions } from './styles/constants'
import { formatDateTime } from './utils/format'

export interface InventoryAlert {
  id: string
  type: 'low_stock' | 'critical_stock' | 'overstock' | 'expiring' | 'expired' | 'anomaly'
  severity: 'info' | 'warning' | 'critical'
  title: string
  message: string
  itemId: string
  itemSku: string
  itemName: string
  currentValue?: number
  thresholdValue?: number
  createdAt: string
  acknowledged: boolean
  acknowledgedBy?: string
  acknowledgedAt?: string
}

export interface InventoryAlertPanelProps {
  alerts: InventoryAlert[]
  onAlertClick?: (alert: InventoryAlert) => void
  onAcknowledge?: (alertId: string) => void
  onDismiss?: (alertId: string) => void
  autoRefresh?: boolean
  refreshInterval?: number
  maxVisible?: number
}

const SEVERITY_STYLES = {
  critical: { bg: 'rgba(239, 68, 68, 0.15)', border: Colors.danger, text: Colors.danger, icon: '🔴' },
  warning: { bg: 'rgba(251, 191, 36, 0.15)', border: Colors.warning, text: Colors.warning, icon: '🟡' },
  info: { bg: 'rgba(59, 130, 246, 0.15)', border: Colors.info, text: Colors.info, icon: '🔵' }
} as const

const TYPE_LABELS = {
  low_stock: '库存偏低', critical_stock: '库存紧急', overstock: '库存超储', expiring: '即将过期', expired: '已过期', anomaly: '数据异常'
} as const

interface AlertCardProps {
  alert: InventoryAlert
  onClick?: () => void
  onAcknowledge?: (id: string) => void
  onDismiss?: (id: string) => void
}

function AlertCard({ alert, onClick, onAcknowledge, onDismiss }: AlertCardProps) {
  const style = SEVERITY_STYLES[alert.severity] || SEVERITY_STYLES.info
  const handleAck = (e: React.MouseEvent) => { e.stopPropagation(); onAcknowledge?.(alert.id) }
  const handleDis = (e: React.MouseEvent) => { e.stopPropagation(); onDismiss?.(alert.id) }
  return (
    <div role="button" tabIndex={0} onClick={onClick} onKeyDown={(e) => e.key === 'Enter' && onClick?.()} style={{ padding: `${Spacing.lg} ${Spacing.xl}`, borderBottom: `1px solid ${Colors.borderLight}`, background: alert.acknowledged ? 'transparent' : style.bg, borderLeft: `3px solid ${style.border}`, cursor: 'pointer', transition: Transitions.normal }} aria-label={`${alert.title} - ${alert.severity}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: Spacing.sm }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: Spacing.md }}>
          <span>{style.icon}</span>
          <span style={{ fontWeight: 500, fontSize: FontSizes.lg }}>{alert.title}</span>
          <span style={{ fontSize: FontSizes.xs, padding: `${Spacing.xs} 6px`, borderRadius: BorderRadius.sm, background: style.bg, color: style.text }}>{TYPE_LABELS[alert.type] || alert.type}</span>
        </div>
        <span style={{ fontSize: FontSizes.sm, color: Colors.textMuted }}>{formatDateTime(alert.createdAt)}</span>
      </div>
      <div style={{ fontSize: FontSizes.md, color: Colors.textSecondary, marginBottom: Spacing.md, marginLeft: '24px' }}>{alert.message}</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginLeft: '24px' }}>
        <div style={{ fontSize: FontSizes.sm, color: Colors.textMuted }}>
          <span style={{ color: Colors.primary }}>{alert.itemSku}</span>
          <span style={{ margin: `0 ${Spacing.sm}` }}>•</span>
          <span>{alert.itemName}</span>
          {alert.currentValue !== undefined && alert.thresholdValue !== undefined && <span style={{ marginLeft: Spacing.md, color: style.text }}>{alert.currentValue} / {alert.thresholdValue}</span>}
        </div>
        <div style={{ display: 'flex', gap: Spacing.md }}>
          {!alert.acknowledged && onAcknowledge && <Button onClick={handleAck} style={{ padding: `${Spacing.xs} 10px` }}>标记已读</Button>}
          {onDismiss && <Button variant="ghost" onClick={handleDis} style={{ padding: `${Spacing.xs} 10px` }}>✕</Button>}
        </div>
      </div>
    </div>
  )
}

export function InventoryAlertPanel({ alerts, onAlertClick, onAcknowledge, onDismiss, autoRefresh = false, refreshInterval = 30000, maxVisible = 10 }: InventoryAlertPanelProps) {
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'acknowledged'>('all')
  const [severityFilter, setSeverityFilter] = useState<'all' | 'critical' | 'warning' | 'info'>('all')
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set())

  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      if (dismissedIds.has(alert.id)) return false
      if (filter === 'unacknowledged' && alert.acknowledged) return false
      if (filter === 'acknowledged' && !alert.acknowledged) return false
      if (severityFilter !== 'all' && alert.severity !== severityFilter) return false
      return true
    }).slice(0, maxVisible)
  }, [alerts, dismissedIds, filter, severityFilter, maxVisible])

  const summary = useMemo(() => ({
    critical: alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length,
    warning: alerts.filter(a => a.severity === 'warning' && !a.acknowledged).length,
    info: alerts.filter(a => a.severity === 'info' && !a.acknowledged).length,
    total: alerts.filter(a => !a.acknowledged).length
  }), [alerts])

  const handleDismiss = (id: string) => { setDismissedIds(prev => new Set([...prev, id])); onDismiss?.(id) }

  return (
    <ErrorBoundary>
      <Card>
        <div style={{ padding: Spacing.xl, borderBottom: `1px solid ${Colors.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: Spacing.lg }}>
            <span style={{ fontWeight: 600, fontSize: FontSizes.xl }}>库存告警</span>
            {summary.total > 0 && <span style={{ background: Colors.danger, color: '#fff', fontSize: FontSizes.xs, fontWeight: 600, padding: '2px 8px', borderRadius: '10px' }}>{summary.total}</span>}
          </div>
          <div style={{ display: 'flex', gap: Spacing.lg, fontSize: FontSizes.md }}>
            <span style={{ color: Colors.danger }}>🔴 {summary.critical}</span>
            <span style={{ color: Colors.warning }}>🟡 {summary.warning}</span>
            <span style={{ color: Colors.info }}>🔵 {summary.info}</span>
          </div>
        </div>
        <div style={{ padding: `${Spacing.lg} ${Spacing.xl}`, borderBottom: `1px solid ${Colors.border}`, display: 'flex', gap: Spacing.md }}>
          {(['all', 'unacknowledged', 'acknowledged'] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)} style={{ padding: `${Spacing.xs} 10px`, borderRadius: BorderRadius.md, border: 'none', background: filter === f ? Colors.border : 'transparent', color: filter === f ? Colors.text : Colors.textMuted, fontSize: FontSizes.sm, cursor: 'pointer', transition: Transitions.normal }}>
              {f === 'all' ? '全部' : f === 'unacknowledged' ? '未处理' : '已处理'}
            </button>
          ))}
          <div style={{ width: '1px', background: Colors.border }} />
          {(['all', 'critical', 'warning', 'info'] as const).map(s => (
            <button key={s} onClick={() => setSeverityFilter(s)} style={{ padding: `${Spacing.xs} 10px`, borderRadius: BorderRadius.md, border: 'none', background: severityFilter === s ? (s === 'critical' ? 'rgba(239, 68, 68, 0.2)' : s === 'warning' ? 'rgba(251, 191, 36, 0.2)' : s === 'info' ? 'rgba(59, 130, 246, 0.2)' : Colors.border) : 'transparent', color: severityFilter === s ? (s === 'critical' ? Colors.danger : s === 'warning' ? Colors.warning : s === 'info' ? Colors.info : Colors.text) : Colors.textMuted, fontSize: FontSizes.sm, cursor: 'pointer', transition: Transitions.normal }}>
              {s === 'all' ? '全部级别' : s === 'critical' ? '紧急' : s === 'warning' ? '警告' : '提示'}
            </button>
          ))}
        </div>
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {filteredAlerts.length === 0 ? <div style={{ padding: '40px', textAlign: 'center', color: Colors.textMuted, fontSize: FontSizes.lg }}>✓ 暂无告警</div> : filteredAlerts.map(alert => <AlertCard key={alert.id} alert={alert} onClick={() => onAlertClick?.(alert)} onAcknowledge={onAcknowledge} onDismiss={handleDismiss} />)}
        </div>
      </Card>
    </ErrorBoundary>
  )
}