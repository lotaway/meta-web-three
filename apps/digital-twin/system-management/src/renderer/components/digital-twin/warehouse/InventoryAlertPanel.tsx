import { useState, useEffect } from 'react'

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

  // Auto refresh
  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => {
      // Trigger refresh callback if needed
    }, refreshInterval)
    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  // Filter alerts
  const filteredAlerts = alerts.filter(alert => {
    if (dismissedIds.has(alert.id)) return false
    if (filter === 'unacknowledged' && alert.acknowledged) return false
    if (filter === 'acknowledged' && !alert.acknowledged) return false
    if (severityFilter !== 'all' && alert.severity !== severityFilter) return false
    return true
  })

  const visibleAlerts = filteredAlerts.slice(0, maxVisible)

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return { bg: 'rgba(239, 68, 68, 0.15)', border: '#ef4444', text: '#ef4444', icon: '🔴' }
      case 'warning': return { bg: 'rgba(251, 191, 36, 0.15)', border: '#fbbf24', text: '#fbbf24', icon: '🟡' }
      case 'info': return { bg: 'rgba(59, 130, 246, 0.15)', border: '#3b82f6', text: '#3b82f6', icon: '🔵' }
      default: return { bg: 'rgba(100, 116, 139, 0.15)', border: '#64748b', text: '#64748b', icon: '⚪' }
    }
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'low_stock': return '库存偏低'
      case 'critical_stock': return '库存紧急'
      case 'overstock': return '库存超储'
      case 'expiring': return '即将过期'
      case 'expired': return '已过期'
      case 'anomaly': return '数据异常'
      default: return type
    }
  }

  const handleDismiss = (alertId: string) => {
    setDismissedIds(prev => new Set([...prev, alertId]))
    onDismiss?.(alertId)
  }

  const handleAcknowledge = (alertId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    onAcknowledge?.(alertId)
  }

  // Group alerts by severity for summary
  const alertSummary = {
    critical: alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length,
    warning: alerts.filter(a => a.severity === 'warning' && !a.acknowledged).length,
    info: alerts.filter(a => a.severity === 'info' && !a.acknowledged).length,
    total: alerts.filter(a => !a.acknowledged).length
  }

  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.9)',
      borderRadius: '12px',
      border: '1px solid #334155',
      color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Header with summary */}
      <div style={{ 
        padding: '16px', 
        borderBottom: '1px solid #334155',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>库存告警</span>
          {alertSummary.total > 0 && (
            <span style={{
              background: '#ef4444',
              color: '#fff',
              fontSize: '11px',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '10px'
            }}>
              {alertSummary.total}
            </span>
          )}
        </div>
        
        {/* Severity summary */}
        <div style={{ display: 'flex', gap: '12px', fontSize: '12px' }}>
          <span style={{ color: '#ef4444' }}>🔴 {alertSummary.critical}</span>
          <span style={{ color: '#fbbf24' }}>🟡 {alertSummary.warning}</span>
          <span style={{ color: '#3b82f6' }}>🔵 {alertSummary.info}</span>
        </div>
      </div>

      {/* Filters */}
      <div style={{ 
        padding: '12px 16px', 
        borderBottom: '1px solid #334155',
        display: 'flex',
        gap: '8px'
      }}>
        {/* Status filter */}
        <div style={{ display: 'flex', gap: '4px' }}>
          {(['all', 'unacknowledged', 'acknowledged'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '4px 10px',
                borderRadius: '4px',
                border: 'none',
                background: filter === f ? '#334155' : 'transparent',
                color: filter === f ? '#f1f5f9' : '#64748b',
                fontSize: '11px',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {f === 'all' ? '全部' : f === 'unacknowledged' ? '未处理' : '已处理'}
            </button>
          ))}
        </div>
        
        <div style={{ width: '1px', background: '#334155' }} />
        
        {/* Severity filter */}
        <div style={{ display: 'flex', gap: '4px' }}>
          {(['all', 'critical', 'warning', 'info'] as const).map(s => (
            <button
              key={s}
              onClick={() => setSeverityFilter(s)}
              style={{
                padding: '4px 10px',
                borderRadius: '4px',
                border: 'none',
                background: severityFilter === s ? 
                  (s === 'critical' ? 'rgba(239, 68, 68, 0.2)' : 
                   s === 'warning' ? 'rgba(251, 191, 36, 0.2)' : 
                   s === 'info' ? 'rgba(59, 130, 246, 0.2)' : '#334155') : 'transparent',
                color: severityFilter === s ? 
                  (s === 'critical' ? '#ef4444' : 
                   s === 'warning' ? '#fbbf24' : 
                   s === 'info' ? '#3b82f6' : '#f1f5f9') : '#64748b',
                fontSize: '11px',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {s === 'all' ? '全部级别' : s === 'critical' ? '紧急' : s === 'warning' ? '警告' : '提示'}
            </button>
          ))}
        </div>
      </div>

      {/* Alert list */}
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {visibleAlerts.length === 0 ? (
          <div style={{ 
            padding: '40px', 
            textAlign: 'center', 
            color: '#64748b',
            fontSize: '13px'
          }}>
            ✓ 暂无告警
          </div>
        ) : (
          visibleAlerts.map(alert => {
            const style = getSeverityColor(alert.severity)
            return (
              <div
                key={alert.id}
                onClick={() => onAlertClick?.(alert)}
                style={{
                  padding: '12px 16px',
                  borderBottom: '1px solid #1e293b',
                  background: alert.acknowledged ? 'transparent' : style.bg,
                  borderLeft: `3px solid ${style.border}`,
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}
              >
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'flex-start',
                  marginBottom: '6px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>{style.icon}</span>
                    <span style={{ fontWeight: 500, fontSize: '13px' }}>{alert.title}</span>
                    <span style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      background: style.bg,
                      color: style.text
                    }}>
                      {getTypeLabel(alert.type)}
                    </span>
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    {new Date(alert.createdAt).toLocaleString('zh-CN')}
                  </span>
                </div>
                
                <div style={{ 
                  fontSize: '12px', 
                  color: '#94a3b8', 
                  marginBottom: '8px',
                  marginLeft: '24px'
                }}>
                  {alert.message}
                </div>
                
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginLeft: '24px'
                }}>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>
                    <span style={{ color: '#38bdf8' }}>{alert.itemSku}</span>
                    <span style={{ margin: '0 8px' }}>•</span>
                    <span>{alert.itemName}</span>
                    {alert.currentValue !== undefined && alert.thresholdValue !== undefined && (
                      <span style={{ marginLeft: '8px', color: style.text }}>
                        {alert.currentValue} / {alert.thresholdValue}
                      </span>
                    )}
                  </div>
                  
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {!alert.acknowledged && onAcknowledge && (
                      <button
                        onClick={(e) => handleAcknowledge(alert.id, e)}
                        style={{
                          padding: '4px 10px',
                          borderRadius: '4px',
                          border: '1px solid #334155',
                          background: 'transparent',
                          color: '#94a3b8',
                          fontSize: '11px',
                          cursor: 'pointer'
                        }}
                      >
                        标记已读
                      </button>
                    )}
                    {onDismiss && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDismiss(alert.id) }}
                        style={{
                          padding: '4px 10px',
                          borderRadius: '4px',
                          border: 'none',
                          background: 'transparent',
                          color: '#64748b',
                          fontSize: '11px',
                          cursor: 'pointer'
                        }}
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Footer */}
      {filteredAlerts.length > maxVisible && (
        <div style={{ 
          padding: '12px 16px', 
          borderTop: '1px solid #334155',
          textAlign: 'center',
          fontSize: '12px',
          color: '#64748b'
        }}>
          还有 {filteredAlerts.length - maxVisible} 条告警未显示
        </div>
      )}
    </div>
  )
}

export type { InventoryAlertPanelProps, InventoryAlert }