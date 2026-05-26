import { useState, useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Button } from './components/Button'
import { Colors, Spacing, FontSizes, BorderRadius, Transitions } from './styles/constants'
import { formatDateTime } from './utils/format'

// ============ 常量提取 ============
const ALERT_CONFIG = {
  maxHeight: '400px',
  badgePadding: '2px 8px',
  badgeRadius: '10px',
  iconMarginLeft: '24px',
  buttonPadding: `${Spacing.xs} 10px`
} as const

const SEVERITY_STYLES = {
  critical: {
    bg: 'rgba(239, 68, 68, 0.15)',
    border: Colors.danger,
    text: Colors.danger,
    icon: '🔴'
  },
  warning: {
    bg: 'rgba(251, 191, 36, 0.15)',
    border: Colors.warning,
    text: Colors.warning,
    icon: '🟡'
  },
  info: {
    bg: 'rgba(59, 130, 246, 0.15)',
    border: Colors.info,
    text: Colors.info,
    icon: '🔵'
  }
} as const

const TYPE_LABELS = {
  low_stock: '库存偏低',
  critical_stock: '库存紧急',
  overstock: '库存超储',
  expiring: '即将过期',
  expired: '已过期',
  anomaly: '数据异常'
} as const

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

// ============ 类型守卫函数 ============
function getSeverityStyle(severity: 'info' | 'warning' | 'critical') {
  return SEVERITY_STYLES[severity] || SEVERITY_STYLES.info
}

function getTypeLabel(type: string): string {
  return TYPE_LABELS[type as keyof typeof TYPE_LABELS] || type
}

// ============ 子组件：告警卡片 ============
interface AlertCardProps {
  alert: InventoryAlert
  onClick?: () => void
  onAcknowledge?: (id: string) => void
  onDismiss?: (id: string) => void
}

function AlertCard({ alert, onClick, onAcknowledge, onDismiss }: AlertCardProps) {
  const style = getSeverityStyle(alert.severity)

  const handleAck = (e: React.MouseEvent) => {
    e.stopPropagation()
    onAcknowledge?.(alert.id)
  }

  const handleDismiss = (e: React.MouseEvent) => {
    e.stopPropagation()
    onDismiss?.(alert.id)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onClick?.()
    }
  }

  const cardStyle = alert.acknowledged
    ? { background: 'transparent' }
    : { background: style.bg }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      style={{
        padding: `${Spacing.lg} ${Spacing.xl}`,
        borderBottom: `1px solid ${Colors.borderLight}`,
        borderLeft: `3px solid ${style.border}`,
        cursor: 'pointer',
        transition: Transitions.normal,
        ...cardStyle
      }}
      aria-label={`${alert.title} - ${alert.severity}`}
    >
      {/* 头部：标题和类型 */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: Spacing.sm
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: Spacing.md }}>
          <span>{style.icon}</span>
          <span style={{ fontWeight: 500, fontSize: FontSizes.lg }}>{alert.title}</span>
          <span
            style={{
              fontSize: FontSizes.xs,
              padding: `${Spacing.xs} 6px`,
              borderRadius: BorderRadius.sm,
              background: style.bg,
              color: style.text
            }}
          >
            {getTypeLabel(alert.type)}
          </span>
        </div>
        <span style={{ fontSize: FontSizes.sm, color: Colors.textMuted }}>
          {formatDateTime(alert.createdAt)}
        </span>
      </div>

      {/* 消息内容 */}
      <div
        style={{
          fontSize: FontSizes.md,
          color: Colors.textSecondary,
          marginBottom: Spacing.md,
          marginLeft: ALERT_CONFIG.iconMarginLeft
        }}
      >
        {alert.message}
      </div>

      {/* 底部：商品信息和操作按钮 */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginLeft: ALERT_CONFIG.iconMarginLeft
        }}
      >
        <div style={{ fontSize: FontSizes.sm, color: Colors.textMuted }}>
          <span style={{ color: Colors.primary }}>{alert.itemSku}</span>
          <span style={{ margin: `0 ${Spacing.sm}` }}>•</span>
          <span>{alert.itemName}</span>
          {alert.currentValue !== undefined && alert.thresholdValue !== undefined && (
            <span style={{ marginLeft: Spacing.md, color: style.text }}>
              {alert.currentValue} / {alert.thresholdValue}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: Spacing.md }}>
          {!alert.acknowledged && onAcknowledge && (
            <Button onClick={handleAck} style={{ padding: ALERT_CONFIG.buttonPadding }}>
              标记已读
            </Button>
          )}
          {onDismiss && (
            <Button variant="ghost" onClick={handleDismiss} style={{ padding: ALERT_CONFIG.buttonPadding }}>
              ✕
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

// ============ 子组件：摘要统计 ============
function AlertSummary({ summary }: { summary: { critical: number; warning: number; info: number; total: number } }) {
  return (
    <div style={{ display: 'flex', gap: Spacing.lg, fontSize: FontSizes.md }}>
      <span style={{ color: Colors.danger }}>🔴 {summary.critical}</span>
      <span style={{ color: Colors.warning }}>🟡 {summary.warning}</span>
      <span style={{ color: Colors.info }}>🔵 {summary.info}</span>
    </div>
  )
}

// ============ 子组件：过滤器按钮组 ============
interface FilterButtonProps {
  label: string
  isActive: boolean
  onClick: () => void
}

function FilterButton({ label, isActive, onClick }: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: `${Spacing.xs} 10px`,
        borderRadius: BorderRadius.md,
        border: 'none',
        background: isActive ? Colors.border : 'transparent',
        color: isActive ? Colors.text : Colors.textMuted,
        fontSize: FontSizes.sm,
        cursor: 'pointer',
        transition: Transitions.normal
      }}
    >
      {label}
    </button>
  )
}

interface FilterBarProps {
  filter: 'all' | 'unacknowledged' | 'acknowledged'
  setFilter: (f: 'all' | 'unacknowledged' | 'acknowledged') => void
  severityFilter: 'all' | 'critical' | 'warning' | 'info'
  setSeverityFilter: (s: 'all' | 'critical' | 'warning' | 'info') => void
}

function FilterBar({ filter, setFilter, severityFilter, setSeverityFilter }: FilterBarProps) {
  const getSeverityButtonStyle = (s: string) => {
    if (severityFilter !== s) return { background: 'transparent' }

    switch (s) {
      case 'critical':
        return { background: 'rgba(239, 68, 68, 0.2)' }
      case 'warning':
        return { background: 'rgba(251, 191, 36, 0.2)' }
      case 'info':
        return { background: 'rgba(59, 130, 246, 0.2)' }
      default:
        return { background: Colors.border }
    }
  }

  const getSeverityTextColor = (s: string) => {
    if (severityFilter !== s) return Colors.textMuted

    switch (s) {
      case 'critical':
        return Colors.danger
      case 'warning':
        return Colors.warning
      case 'info':
        return Colors.info
      default:
        return Colors.text
    }
  }

  return (
    <div
      style={{
        padding: `${Spacing.lg} ${Spacing.xl}`,
        borderBottom: `1px solid ${Colors.border}`,
        display: 'flex',
        gap: Spacing.md
      }}
    >
      {/* 状态过滤器 */}
      <FilterButton
        label="全部"
        isActive={filter === 'all'}
        onClick={() => setFilter('all')}
      />
      <FilterButton
        label="未处理"
        isActive={filter === 'unacknowledged'}
        onClick={() => setFilter('unacknowledged')}
      />
      <FilterButton
        label="已处理"
        isActive={filter === 'acknowledged'}
        onClick={() => setFilter('acknowledged')}
      />

      <div style={{ width: '1px', background: Colors.border }} />

      {/* 严重级别过滤器 */}
      <FilterButton
        label="全部级别"
        isActive={severityFilter === 'all'}
        onClick={() => setSeverityFilter('all')}
      />
      <button
        onClick={() => setSeverityFilter('critical')}
        style={{
          padding: `${Spacing.xs} 10px`,
          borderRadius: BorderRadius.md,
          border: 'none',
          ...getSeverityButtonStyle('critical'),
          color: getSeverityTextColor('critical'),
          fontSize: FontSizes.sm,
          cursor: 'pointer',
          transition: Transitions.normal
        }}
      >
        紧急
      </button>
      <button
        onClick={() => setSeverityFilter('warning')}
        style={{
          padding: `${Spacing.xs} 10px`,
          borderRadius: BorderRadius.md,
          border: 'none',
          ...getSeverityButtonStyle('warning'),
          color: getSeverityTextColor('warning'),
          fontSize: FontSizes.sm,
          cursor: 'pointer',
          transition: Transitions.normal
        }}
      >
        警告
      </button>
      <button
        onClick={() => setSeverityFilter('info')}
        style={{
          padding: `${Spacing.xs} 10px`,
          borderRadius: BorderRadius.md,
          border: 'none',
          ...getSeverityButtonStyle('info'),
          color: getSeverityTextColor('info'),
          fontSize: FontSizes.sm,
          cursor: 'pointer',
          transition: Transitions.normal
        }}
      >
        提示
      </button>
    </div>
  )
}

// ============ 子组件：告警列表 ============
function AlertList({
  alerts,
  onAlertClick,
  onAcknowledge,
  onDismiss,
  onHandleDismiss
}: {
  alerts: InventoryAlert[]
  onAlertClick?: (alert: InventoryAlert) => void
  onAcknowledge?: (alertId: string) => void
  onDismiss?: (alertId: string) => void
  onHandleDismiss: (id: string) => void
}) {
  if (alerts.length === 0) {
    return (
      <div
        style={{
          padding: '40px',
          textAlign: 'center',
          color: Colors.textMuted,
          fontSize: FontSizes.lg
        }}
      >
        ✓ 暂无告警
      </div>
    )
  }

  return (
    <>
      {alerts.map(alert => (
        <AlertCard
          key={alert.id}
          alert={alert}
          onClick={() => onAlertClick?.(alert)}
          onAcknowledge={onAcknowledge}
          onDismiss={onHandleDismiss}
        />
      ))}
    </>
  )
}

// ============ 主组件 ============
export function InventoryAlertPanel({
  alerts,
  onAlertClick,
  onAcknowledge,
  onDismiss,
  autoRefresh = false,
  refreshInterval = 30000,
  maxVisible = 10
}: InventoryAlertPanelProps) {
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'acknowledged'>('all')
  const [severityFilter, setSeverityFilter] = useState<'all' | 'critical' | 'warning' | 'info'>('all')
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set())

  // 过滤后的告警列表
  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      if (dismissedIds.has(alert.id)) return false

      if (filter === 'unacknowledged' && alert.acknowledged) return false
      if (filter === 'acknowledged' && !alert.acknowledged) return false

      if (severityFilter !== 'all' && alert.severity !== severityFilter) return false

      return true
    }).slice(0, maxVisible)
  }, [alerts, dismissedIds, filter, severityFilter, maxVisible])

  // 摘要统计
  const summary = useMemo(() => ({
    critical: alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length,
    warning: alerts.filter(a => a.severity === 'warning' && !a.acknowledged).length,
    info: alerts.filter(a => a.severity === 'info' && !a.acknowledged).length,
    total: alerts.filter(a => !a.acknowledged).length
  }), [alerts])

  // 处理关闭告警
  const handleDismiss = (id: string) => {
    setDismissedIds(prev => new Set([...prev, id]))
    onDismiss?.(id)
  }

  // 头部信息
  const header = (
    <div
      style={{
        padding: Spacing.xl,
        borderBottom: `1px solid ${Colors.border}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: Spacing.lg }}>
        <span style={{ fontWeight: 600, fontSize: FontSizes.xl }}>库存告警</span>
        {summary.total > 0 && (
          <span
            style={{
              background: Colors.danger,
              color: '#fff',
              fontSize: FontSizes.xs,
              fontWeight: 600,
              padding: ALERT_CONFIG.badgePadding,
              borderRadius: ALERT_CONFIG.badgeRadius
            }}
          >
            {summary.total}
          </span>
        )}
      </div>
      <AlertSummary summary={summary} />
    </div>
  )

  return (
    <ErrorBoundary>
      <Card>
        {header}
        <FilterBar
          filter={filter}
          setFilter={setFilter}
          severityFilter={severityFilter}
          setSeverityFilter={setSeverityFilter}
        />
        <div style={{ maxHeight: ALERT_CONFIG.maxHeight, overflowY: 'auto' }}>
          <AlertList
            alerts={filteredAlerts}
            onAlertClick={onAlertClick}
            onAcknowledge={onAcknowledge}
            onDismiss={onDismiss}
            onHandleDismiss={handleDismiss}
          />
        </div>
      </Card>
    </ErrorBoundary>
  )
}