import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { API_BASE_URL } from '@/api/generated'

interface LogisticsTrace {
  id: number
  time: string
  content: string
  status?: number
}

export default function LogisticsScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  const orderId = params.id

  const [traces, setTraces] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [logisticsInfo, setLogisticsInfo] = useState({
    company: '',
    trackingNo: '',
    phone: '',
  })

  useEffect(() => {
    loadLogistics()
  }, [orderId])

  const loadLogistics = async () => {
    if (!orderId) return
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/order-service/order/${orderId}/logistics`)
      const result = await response.json()
      if (result.data) {
        setLogisticsInfo({
          company: result.data.company || '',
          trackingNo: result.data.trackingNo || '',
          phone: result.data.phone || '',
        })
        setTraces(result.data.traces || [])
      }
    } catch (error) {
      console.error('Failed to load logistics:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCallDelivery = () => {
    console.log('Call:', logisticsInfo.phone)
  }

  const handleCopyTrackingNo = () => {
    console.log('Copy:', logisticsInfo.trackingNo)
  }

  const renderTraceItem = ({ item, index }: { item: LogisticsTrace; index: number }) => {
    const isLatest = index === 0
    return (
      <View style={styles.traceItem}>
        <View style={styles.traceTimeline}>
          <View style={[
            styles.timelineDot,
            isLatest ? styles.timelineDotActive : styles.timelineDotNormal,
            isLatest ? { backgroundColor: colors.primary } : { backgroundColor: colors.textSecondary },
          ]} />
          {index < traces.length - 1 && (
            <View style={[styles.timelineLine, { backgroundColor: colors.border }]} />
          )}
        </View>
        <View style={styles.traceContent}>
          <Text style={[
            styles.traceTime,
            { color: isLatest ? colors.primary : colors.textSecondary },
          ]}>
            {item.time}
          </Text>
          <Text style={[
            styles.traceText,
            { color: isLatest ? colors.text : colors.textSecondary },
          ]}>
            {item.content}
          </Text>
        </View>
      </View>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('logistics.title')}</Text>
        <View style={styles.headerRight} />
      </View>

      {/* 物流信息 */}
      <View style={[styles.logisticsInfo, { backgroundColor: colors.card, borderBottomColor: colors.border }]}>
        <View style={styles.logisticsRow}>
          <IconSymbol name="shippingbox" size={20} color={colors.primary} />
          <Text style={[styles.logisticsLabel, { color: colors.textSecondary }]}>{t('logistics.company')}</Text>
          <Text style={[styles.logisticsValue, { color: colors.text }]}>{logisticsInfo.company}</Text>
        </View>
        <View style={styles.logisticsRow}>
          <IconSymbol name="doc" size={20} color={colors.primary} />
          <Text style={[styles.logisticsLabel, { color: colors.textSecondary }]}>{t('logistics.tracking_no')}</Text>
          <Text style={[styles.logisticsValue, { color: colors.text }]}>{logisticsInfo.trackingNo}</Text>
          <TouchableOpacity style={styles.copyBtn} onPress={handleCopyTrackingNo}>
            <Text style={[styles.copyBtnText, { color: colors.primary }]}>{t('logistics.copy')}</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.logisticsRow}>
          <IconSymbol name="phone" size={20} color={colors.primary} />
          <Text style={[styles.logisticsLabel, { color: colors.textSecondary }]}>{t('logistics.phone')}</Text>
          <Text style={[styles.logisticsValue, { color: colors.text }]}>{logisticsInfo.phone}</Text>
          <TouchableOpacity style={styles.copyBtn} onPress={handleCallDelivery}>
            <Text style={[styles.copyBtnText, { color: colors.primary }]}>{t('logistics.call')}</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* 物流轨迹 */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : traces.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="shippingbox" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('logistics.no_traces')}</Text>
        </View>
      ) : (
        <FlatList
          data={traces}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderTraceItem}
          contentContainerStyle={styles.tracesContent}
        />
      )}
    </SafeAreaView>
  )
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
