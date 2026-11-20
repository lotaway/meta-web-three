import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { API_BASE_URL } from '@/api/generated'

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type OrderId = ReturnType<typeof useLocalSearchParams>['id']
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetTraces = React.Dispatch<React.SetStateAction<any[]>>

interface LogisticsTrace {
  id: number
  time: string
  content: string
  status?: number
}

interface LogisticsInfo {
  company: string
  trackingNo: string
  phone: string
}

type SetLogisticsInfo = React.Dispatch<React.SetStateAction<LogisticsInfo>>

function mapLogisticsInfo(data: any): LogisticsInfo {
  return {
    company: data.company || '',
    trackingNo: data.trackingNo || '',
    phone: data.phone || '',
  }
}

async function loadLogisticsData(orderId: OrderId, setTraces: SetTraces, setLogisticsInfo: SetLogisticsInfo, setLoading: SetBoolean): Promise<void> {
  if (!orderId) return
  setLoading(true)
  try {
    const response = await fetch(`${API_BASE_URL}/mall-trade-service/order/${orderId}/logistics`)
    const result = await response.json()
    if (result.data) {
      setLogisticsInfo(mapLogisticsInfo(result.data))
      setTraces(result.data.traces || [])
    }
  } catch (error) {
    console.error('Failed to load logistics:', error)
    Alert.alert('Error', 'Failed to load the logistics.')
  } finally {
    setLoading(false)
  }
}

function useLogisticsState() {
  const [traces, setTraces] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [logisticsInfo, setLogisticsInfo] = useState<LogisticsInfo>({ company: '', trackingNo: '', phone: '' })
  return { traces, setTraces, loading, setLoading, logisticsInfo, setLogisticsInfo }
}

type LogisticsState = ReturnType<typeof useLogisticsState>

function createLogisticsActions(router: AppRouter, state: LogisticsState) {
  return {
    onBack: () => router.back(),
    onCopyTrackingNo: () => console.log('Copy:', state.logisticsInfo.trackingNo),
    onCallDelivery: () => console.log('Call:', state.logisticsInfo.phone),
  }
}

type LogisticsActions = ReturnType<typeof createLogisticsActions>

function useLogisticsController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  const orderId = params.id
  const state = useLogisticsState()
  const loadLogistics = () => loadLogisticsData(orderId, state.setTraces, state.setLogisticsInfo, state.setLoading)
  useEffect(() => { loadLogistics() }, [orderId])
  return { t, colors, ...state, ...createLogisticsActions(router, state) }
}

interface LogisticsViewProps extends LogisticsActions {
  colors: AppColors
  t: Translate
  traces: any[]
  loading: boolean
  logisticsInfo: LogisticsInfo
}

function LogisticsHeader({ colors, t, onBack }: Pick<LogisticsViewProps, 'colors' | 't' | 'onBack'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{t('logistics.title')}</Text>
      <View style={styles.headerRight} />
    </View>
  )
}

interface LogisticsInfoRowProps {
  icon: 'shippingbox' | 'doc' | 'phone'
  label: string
  value: string
  actionLabel?: string
  onAction?: () => void
  colors: AppColors
}

function LogisticsInfoRow({ icon, label, value, actionLabel, onAction, colors }: LogisticsInfoRowProps) {
  return (
    <View style={styles.logisticsRow}>
      <IconSymbol name={icon} size={20} color={colors.primary} />
      <Text style={[styles.logisticsLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.logisticsValue, { color: colors.text }]}>{value}</Text>
      {!!actionLabel && (
        <TouchableOpacity style={styles.copyBtn} onPress={onAction}>
          <Text style={[styles.copyBtnText, { color: colors.primary }]}>{actionLabel}</Text>
        </TouchableOpacity>
      )}
    </View>
  )
}

function LogisticsInfoCard({ colors, t, logisticsInfo, onCopyTrackingNo, onCallDelivery }: LogisticsViewProps) {
  return (
    <View style={[styles.logisticsInfo, { backgroundColor: colors.card, borderBottomColor: colors.border }]}>
      <LogisticsInfoRow icon="shippingbox" label={t('logistics.company')} value={logisticsInfo.company} colors={colors} />
      <LogisticsInfoRow icon="doc" label={t('logistics.tracking_no')} value={logisticsInfo.trackingNo} actionLabel={t('logistics.copy')} onAction={onCopyTrackingNo} colors={colors} />
      <LogisticsInfoRow icon="phone" label={t('logistics.phone')} value={logisticsInfo.phone} actionLabel={t('logistics.call')} onAction={onCallDelivery} colors={colors} />
    </View>
  )
}

function LoadingState({ colors }: { colors: AppColors }) {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color={colors.primary} />
    </View>
  )
}

function EmptyTraces({ colors, t }: { colors: AppColors; t: Translate }) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="shippingbox" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('logistics.no_traces')}</Text>
    </View>
  )
}

function TraceTimeline({ index, total, colors }: { index: number; total: number; colors: AppColors }) {
  const isLatest = index === 0
  return (
    <View style={styles.traceTimeline}>
      <View style={[
        styles.timelineDot,
        isLatest ? styles.timelineDotActive : styles.timelineDotNormal,
        isLatest ? { backgroundColor: colors.primary } : { backgroundColor: colors.textSecondary },
      ]} />
      {index < total - 1 && <View style={[styles.timelineLine, { backgroundColor: colors.border }]} />}
    </View>
  )
}

function TraceItem({ item, index, total, colors }: { item: LogisticsTrace; index: number; total: number; colors: AppColors }) {
  const isLatest = index === 0
  return (
    <View style={styles.traceItem}>
      <TraceTimeline index={index} total={total} colors={colors} />
      <View style={styles.traceContent}>
        <Text style={[styles.traceTime, { color: isLatest ? colors.primary : colors.textSecondary }]}>
          {item.time}
        </Text>
        <Text style={[styles.traceText, { color: isLatest ? colors.text : colors.textSecondary }]}>
          {item.content}
        </Text>
      </View>
    </View>
  )
}

function TraceList({ traces, colors }: { traces: any[]; colors: AppColors }) {
  return (
    <FlatList
      data={traces}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item, index }) => (
        <TraceItem item={item} index={index} total={traces.length} colors={colors} />
      )}
      contentContainerStyle={styles.tracesContent}
    />
  )
}

function LogisticsView(props: LogisticsViewProps) {
  const { colors, traces, loading } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <LogisticsHeader {...props} />
      <LogisticsInfoCard {...props} />
      {loading ? (
        <LoadingState colors={colors} />
      ) : traces.length === 0 ? (
        <EmptyTraces colors={colors} t={props.t} />
      ) : (
        <TraceList traces={traces} colors={colors} />
      )}
    </SafeAreaView>
  )
}

export default function LogisticsScreen() {
  const controller = useLogisticsController()
  return <LogisticsView {...controller} />
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  backBtn: { padding: 8 },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginRight: 40,
  },
  headerRight: { width: 40 },
  logisticsInfo: {
    padding: 16,
    borderBottomWidth: 1,
  },
  logisticsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    gap: 8,
  },
  logisticsLabel: {
    fontSize: 14,
    width: 70,
  },
  logisticsValue: {
    flex: 1,
    fontSize: 14,
  },
  copyBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderWidth: 1,
    borderRadius: 4,
    borderColor: '#333',
  },
  copyBtnText: {
    fontSize: 12,
  },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyText: { marginTop: 12, fontSize: 14 },
  tracesContent: {
    padding: 16,
  },
  traceItem: {
    flexDirection: 'row',
    paddingBottom: 24,
  },
  traceTimeline: {
    alignItems: 'center',
    marginRight: 12,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  timelineDotActive: {
    width: 14,
    height: 14,
    borderRadius: 7,
  },
  timelineDotNormal: {
    opacity: 0.5,
  },
  timelineLine: {
    width: 2,
    flex: 1,
    marginTop: 4,
  },
  traceContent: {
    flex: 1,
    paddingTop: 2,
  },
  traceTime: {
    fontSize: 12,
    marginBottom: 4,
  },
  traceText: {
    fontSize: 14,
    lineHeight: 20,
  },
})
