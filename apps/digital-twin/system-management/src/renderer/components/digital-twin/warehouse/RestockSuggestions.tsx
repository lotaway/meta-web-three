import { useState } from 'react'

export interface RestockSuggestion {
  id: string
  itemId: string
  itemSku: string
  itemName: string
  category: string
  currentStock: number
  suggestedQuantity: number
  urgency: 'critical' | 'high' | 'medium' | 'low'
  reason: string
  estimatedArrival?: string
  confidence: number // 0-100
  createdAt: string
  status: 'pending' | 'approved' | 'ordered' | 'rejected'
  approver?: string
  approvedAt?: string
}

export interface RestockSuggestionsProps {
  suggestions: RestockSuggestion[]
  onApprove?: (suggestionId: string) => void
  onReject?: (suggestionId: string, reason?: string) => void
  onOrder?: (suggestionId: string) => void
  onDetailsClick?: (suggestion: RestockSuggestion) => void
  groupByUrgency?: boolean
}

export function RestockSuggestions({ 
  suggestions, 
  onApprove,
  onReject,
  onOrder,
  onDetailsClick,
  groupByUrgency = true
}: RestockSuggestionsProps) {
  const [filter, setFilter] = useState<'all' | 'pending' | 'approved' | 'ordered'>('all')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const filteredSuggestions = suggestions.filter(s => {
    if (filter === 'all') return true
    return s.status === filter
  })

  const groupedSuggestions = groupByUrgency 
    ? {
        critical: filteredSuggestions.filter(s => s.urgency === 'critical'),
        high: filteredSuggestions.filter(s => s.urgency === 'high'),
        medium: filteredSuggestions.filter(s => s.urgency === 'medium'),
        low: filteredSuggestions.filter(s => s.urgency === 'low')
      }
    : { all: filteredSuggestions }

  const getUrgencyStyle = (urgency: string) => {
    switch (urgency) {
      case 'critical': return { bg: 'rgba(239, 68, 68, 0.15)', border: '#ef4444', label: '紧急', labelColor: '#ef4444' }
      case 'high': return { bg: 'rgba(251, 191, 36, 0.15)', border: '#fbbf24', label: '高', labelColor: '#fbbf24' }
      case 'medium': return { bg: 'rgba(59, 130, 246, 0.15)', border: '#3b82f6', label: '中', labelColor: '#3b82f6' }
      case 'low': return { bg: 'rgba(100, 116, 139, 0.15)', border: '#64748b', label: '低', labelColor: '#64748b' }
      default: return { bg: 'rgba(100, 116, 139, 0.15)', border: '#64748b', label: '低', labelColor: '#64748b' }
    }
  }

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'pending': return { bg: 'rgba(251, 191, 36, 0.2)', text: '#fbbf24' }
      case 'approved': return { bg: 'rgba(34, 197, 94, 0.2)', text: '#22c55e' }
      case 'ordered': return { bg: 'rgba(59, 130, 246, 0.2)', text: '#3b82f6' }
      case 'rejected': return { bg: 'rgba(239, 68, 68, 0.2)', text: '#ef4444' }
      default: return { bg: 'rgba(100, 116, 139, 0.2)', text: '#64748b' }
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending': return '待审批'
      case 'approved': return '已审批'
      case 'ordered': return '已下单'
      case 'rejected': return '已拒绝'
      default: return status
    }
  }

  const urgencyOrder = ['critical', 'high', 'medium', 'low']

  const renderSuggestion = (suggestion: RestockSuggestion) => {
    const urgencyStyle = getUrgencyStyle(suggestion.urgency)
    const statusStyle = getStatusStyle(suggestion.status)
    const isExpanded = expandedId === suggestion.id

    return (
      <div
        key={suggestion.id}
        style={{
          background: isExpanded ? 'rgba(30, 41, 59, 0.9)' : 'rgba(15, 23, 42, 0.6)',
          borderRadius: '8px',
          border: `1px solid ${isExpanded ? '#38bdf8' : '#334155'}`,
          marginBottom: '8px',
          overflow: 'hidden',
          transition: 'all 0.2s'
        }}
      >
        {/* Main content */}
        <div 
          onClick={() => setExpandedId(isExpanded ? null : suggestion.id)}
          style={{
            padding: '12px 16px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}
        >
          {/* Urgency indicator */}
          <div style={{
            width: '4px',
            height: '40px',
            background: urgencyStyle.border,
            borderRadius: '2px'
          }} />

          {/* Item info */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span style={{ fontWeight: 500, fontSize: '13px', color: '#f1f5f9' }}>
                {suggestion.itemName}
              </span>
              <span style={{ 
                fontSize: '10px', 
                padding: '2px 6px', 
                borderRadius: '3px',
                background: urgencyStyle.bg,
                color: urgencyStyle.labelColor,
                fontWeight: 500
              }}>
                {urgencyStyle.label}
              </span>
              <span style={{ 
                fontSize: '10px', 
                padding: '2px 6px', 
                borderRadius: '3px',
                background: statusStyle.bg,
                color: statusStyle.text,
                fontWeight: 500
              }}>
                {getStatusLabel(suggestion.status)}
              </span>
            </div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>
              <span style={{ color: '#38bdf8' }}>{suggestion.itemSku}</span>
              <span style={{ margin: '0 6px' }}>•</span>
              <span>{suggestion.category}</span>
            </div>
          </div>

          {/* Quantity and confidence */}
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '18px', fontWeight: 600, color: '#f1f5f9' }}>
              +{suggestion.suggestedQuantity}
            </div>
            <div style={{ fontSize: '10px', color: '#64748b' }}>
              当前: {suggestion.currentStock}
            </div>
          </div>

          {/* Expand icon */}
          <div style={{ 
            color: '#64748b', 
            fontSize: '12px',
            transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }}>
            ▼
          </div>
        </div>

        {/* Expanded details */}
        {isExpanded && (
          <div style={{
            padding: '12px 16px',
            borderTop: '1px solid #334155',
            background: 'rgba(15, 23, 42, 0.8)'
          }}>
            <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '12px' }}>
              {suggestion.reason}
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(2, 1fr)', 
              gap: '8px',
              fontSize: '11px',
              marginBottom: '12px'
            }}>
              <div>
                <span style={{ color: '#64748b' }}>置信度: </span>
                <span style={{ color: suggestion.confidence >= 80 ? '#22c55e' : 
                               suggestion.confidence >= 60 ? '#fbbf24' : '#ef4444' }}>
                  {suggestion.confidence}%
                </span>
              </div>
              {suggestion.estimatedArrival && (
                <div>
                  <span style={{ color: '#64748b' }}>预计到货: </span>
                  <span style={{ color: '#f1f5f9' }}>{suggestion.estimatedArrival}</span>
                </div>
              )}
              <div>
                <span style={{ color: '#64748b' }}>创建时间: </span>
                <span style={{ color: '#94a3b8' }}>
                  {new Date(suggestion.createdAt).toLocaleString('zh-CN')}
                </span>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {suggestion.status === 'pending' && onApprove && (
                <button
                  onClick={(e) => { e.stopPropagation(); onApprove(suggestion.id) }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: 'none',
                    background: '#22c55e',
                    color: '#fff',
                    fontSize: '11px',
                    fontWeight: 500,
                    cursor: 'pointer'
                  }}
                >
                  审批通过
                </button>
              )}
              {suggestion.status === 'pending' && onReject && (
                <button
                  onClick={(e) => { e.stopPropagation(); onReject(suggestion.id) }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: '1px solid #334155',
                    background: 'transparent',
                    color: '#94a3b8',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}
                >
                  拒绝
                </button>
              )}
              {suggestion.status === 'approved' && onOrder && (
                <button
                  onClick={(e) => { e.stopPropagation(); onOrder(suggestion.id) }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: 'none',
                    background: '#3b82f6',
                    color: '#fff',
                    fontSize: '11px',
                    fontWeight: 500,
                    cursor: 'pointer'
                  }}
                >
                  下单
                </button>
              )}
              {onDetailsClick && (
                <button
                  onClick={(e) => { e.stopPropagation(); onDetailsClick(suggestion) }}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: '1px solid #334155',
                    background: 'transparent',
                    color: '#94a3b8',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}
                >
                  查看详情
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    )
  }

  const urgencyLabels: Record<string, string> = {
    critical: '紧急',
    high: '高优先级',
    medium: '中等',
    low: '低优先级',
    all: '全部'
  }

  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.9)',
      borderRadius: '12px',
      border: '1px solid #334155',
      color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Header */}
      <div style={{ 
        padding: '16px', 
        borderBottom: '1px solid #334155',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>补货建议</span>
          <span style={{
            background: 'rgba(239, 68, 68, 0.2)',
            color: '#ef4444',
            fontSize: '11px',
            fontWeight: 600,
            padding: '2px 8px',
            borderRadius: '10px'
          }}>
            {filteredSuggestions.length}
          </span>
        </div>
        
        {/* Filter buttons */}
        <div style={{ display: 'flex', gap: '4px' }}>
          {(['all', 'pending', 'approved', 'ordered'] as const).map(f => (
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
                cursor: 'pointer'
              }}
            >
              {f === 'all' ? '全部' : 
               f === 'pending' ? '待审批' : 
               f === 'approved' ? '已审批' : '已下单'}
            </button>
          ))}
        </div>
      </div>

      {/* Suggestions list */}
      <div style={{ maxHeight: '500px', overflowY: 'auto', padding: '16px' }}>
        {groupByUrgency ? (
          urgencyOrder.map(urgency => {
            const group = groupedSuggestions[urgency as keyof typeof groupedSuggestions]
            if (!group || group.length === 0) return null
            
            return (
              <div key={urgency} style={{ marginBottom: '16px' }}>
                <div style={{ 
                  fontSize: '11px', 
                  color: getUrgencyStyle(urgency).labelColor,
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span style={{ 
                    width: '8px', 
                    height: '8px', 
                    borderRadius: '50%', 
                    background: getUrgencyStyle(urgency).border 
                  }} />
                  {urgencyLabels[urgency]} ({group.length})
                </div>
                {group.map(renderSuggestion)}
              </div>
            )
          })
        ) : (
          filteredSuggestions.map(renderSuggestion)
        )}
        
        {filteredSuggestions.length === 0 && (
          <div style={{ 
            padding: '40px', 
            textAlign: 'center', 
            color: '#64748b',
            fontSize: '13px'
          }}>
            暂无补货建议
          </div>
        )}
      </div>
    </div>
  )
}

export type { RestockSuggestionsProps, RestockSuggestion }