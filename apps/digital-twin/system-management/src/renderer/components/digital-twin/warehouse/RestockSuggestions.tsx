import { useState, useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { FilterTabs } from './components/FilterTabs'
import { SuggestionCard } from './components/SuggestionCard'
import { Colors, Spacing, FontSizes } from './styles/constants'

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
  confidence: number
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

const URGENCY_ORDER = ['critical', 'high', 'medium', 'low'] as const

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

  const filteredSuggestions = useMemo(() => {
    return suggestions.filter(s => filter === 'all' || s.status === filter)
  }, [suggestions, filter])

  const groupedSuggestions = useMemo(() => {
    if (!groupByUrgency) return { all: filteredSuggestions }
    const groups = { critical: [] as RestockSuggestion[], high: [] as RestockSuggestion[], medium: [] as RestockSuggestion[], low: [] as RestockSuggestion[] }
    filteredSuggestions.forEach(s => {
      if (s.urgency === 'critical') groups.critical.push(s)
      else if (s.urgency === 'high') groups.high.push(s)
      else if (s.urgency === 'medium') groups.medium.push(s)
      else groups.low.push(s)
    })
    return groups
  }, [filteredSuggestions, groupByUrgency])

  const renderGroup = (urgency: string, items: RestockSuggestion[]) => {
    if (items.length === 0) return null
    const color = urgency === 'critical' ? Colors.danger : urgency === 'high' ? Colors.warning : urgency === 'medium' ? Colors.info : Colors.textMuted
    return (
      <div key={urgency} style={{ marginBottom: Spacing.xl }}>
        <div style={{ fontSize: FontSizes.xs, color, marginBottom: Spacing.md, display: 'flex', alignItems: 'center', gap: Spacing.xs }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
          {urgency === 'critical' ? '紧急' : urgency === 'high' ? '高优先级' : urgency === 'medium' ? '中等' : '低优先级'} ({items.length})
        </div>
        {items.map(s => (
          <SuggestionCard key={s.id} suggestion={s} isExpanded={expandedId === s.id} onToggle={() => setExpandedId(expandedId === s.id ? null : s.id)} onApprove={onApprove} onReject={onReject} onOrder={onOrder} onDetailsClick={onDetailsClick} />
        ))}
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <Card style={{ padding: Spacing.xl }}>
        <div style={{ paddingBottom: Spacing.lg, borderBottom: `1px solid ${Colors.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FilterTabs activeFilter={filter} onChange={setFilter} count={filteredSuggestions.length} />
        </div>
        <div style={{ maxHeight: '500px', overflowY: 'auto', padding: Spacing.xl }}>
          {groupByUrgency ? URGENCY_ORDER.map(u => renderGroup(u, groupedSuggestions[u as keyof typeof groupedSuggestions] || [])) : filteredSuggestions.map(s => <SuggestionCard key={s.id} suggestion={s} isExpanded={expandedId === s.id} onToggle={() => setExpandedId(expandedId === s.id ? null : s.id)} onApprove={onApprove} onReject={onReject} onOrder={onOrder} onDetailsClick={onDetailsClick} />)}
          {filteredSuggestions.length === 0 && <div style={{ padding: '40px', textAlign: 'center', color: Colors.textMuted, fontSize: FontSizes.lg }}>暂无补货建议</div>}
        </div>
      </Card>
    </ErrorBoundary>
  )
}