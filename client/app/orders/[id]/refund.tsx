import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  Modal,
  FlatList,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

const REFUND_TYPES = [
  { value: 1, label: '退款退货' },
  { value: 2, label: '仅退款' },
]

const REFUND_REASONS = [
  { value: 1, label: '不想要了' },
  { value: 2, label: '商品损坏' },
  { value: 3, label: '与描述不符' },
  { value: 4, label: '质量问题' },
  { value: 5, label: '发错货' },
  { value: 6, label: '其他原因' },
]

const REFUND_STATUS_MAP: Record<number, { text: string; color: string }> = {
  0: { text: '待处理', color: '#FF6B35' },
  1: { text: '商家已同意', color: '#5fcda2' },
  2: { text: '商家已拒绝', color: '#FF3B30' },
  3: { text: '退款成功', color: '#4A90E2' },
  4: { text: '已取消', color: '#8E8E93' },
}

export default function RefundScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  const orderId = params.id

  const [order, setOrder] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const [refundType, setRefundType] = useState(1)
  const [refundAmount, setRefundAmount] = useState('')
  const [selectedReason, setSelectedReason] = useState<number | null>(null)
  const [description, setDescription] = useState('')

  const [showHistory, setShowHistory] = useState(false)
  const [refundHistory, setRefundHistory] = useState<any[]>([])

  useEffect(() => {
    loadOrder()
    loadRefundHistory()
  }, [orderId])

  const loadOrder = async () => {
    setLoading(true)
    try {
      // TODO: 调用订单详情API
      await new Promise(resolve => setTimeout(resolve, 500))
      const mockOrder = {
        id: orderId,
        payAmount: 299.00,
        orderItems: [
          { id: 1, productName: '测试商品', productPic: '', productPrice: 299, productQuantity: 1 },
        ],
      }
      setOrder(mockOrder)
      setRefundAmount('299.00')
    } catch (error) {
      console.error('Failed to load order:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadRefundHistory = async () => {
    try {
      // TODO: 调用退款历史API
      setRefundHistory([])
    } catch (error) {
      console.error('Failed to load refund history:', error)
    }
  }

  const handleSubmit = async () => {
    if (!selectedReason) {
      Alert.alert(t('common.error'), t('refund.reason_required'))
      return
    }
    if (!refundAmount || parseFloat(refundAmount) <= 0) {
      Alert.alert(t('common.error'), t('refund.amount_required'))
      return
    }

    setSubmitting(true)
    try {
      // TODO: 调用退款申请API
      await new Promise(resolve => setTimeout(resolve, 1000))
      Alert.alert(t('common.success'), t('refund.submit_success'), [
        { text: 'OK', onPress: () => router.back() },
      ])
    } catch (error) {
      Alert.alert(t('common.error'), t('refund.submit_failed'))
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('refund.title')}</Text>
        <TouchableOpacity style={styles.historyBtn} onPress={() => setShowHistory(true)}>
          <IconSymbol name="clock.fill" size={20} color={colors.primary} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {/* 退款类型 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.type_title')}</Text>
          <View style={styles.typeRow}>
            {REFUND_TYPES.map(type => (
              <TouchableOpacity
                key={type.value}
                style={[
                  styles.typeBtn,
                  { borderColor: refundType === type.value ? colors.primary : colors.border },
                  refundType === type.value && { backgroundColor: colors.primary + '10' },
                ]}
                onPress={() => setRefundType(type.value)}
              >
                <Text
                  style={[
                    styles.typeBtnText,
                    { color: refundType === type.value ? colors.primary : colors.textSecondary },
                  ]}
                >
                  {type.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* 退款原因 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.reason_title')}</Text>
          <View style={styles.reasonGrid}>
            {REFUND_REASONS.map(reason => (
              <TouchableOpacity
                key={reason.value}
                style={[
                  styles.reasonBtn,
                  {
                    borderColor: selectedReason === reason.value ? colors.primary : colors.border,
                    backgroundColor: selectedReason === reason.value ? colors.primary + '10' : 'transparent',
                  },
                ]}
                onPress={() => setSelectedReason(reason.value)}
              >
                <Text
                  style={[
                    styles.reasonBtnText,
                    { color: selectedReason === reason.value ? colors.primary : colors.text },
                  ]}
                >
                  {reason.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* 退款金额 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.amount_title')}</Text>
          <View style={styles.amountRow}>
            <Text style={[styles.amountSymbol, { color: colors.primary }]}>¥</Text>
            <TextInput
              style={[styles.amountInput, { color: colors.primary }]}
              value={refundAmount}
              onChangeText={setRefundAmount}
              keyboardType="decimal-pad"
              placeholder="0.00"
              placeholderTextColor={colors.textSecondary}
            />
          </View>
          <Text style={[styles.amountHint, { color: colors.textSecondary }]}>
            {t('refund.amount_hint')}: ¥{order?.payAmount}
          </Text>
        </View>

        {/* 退款说明 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.description_title')}</Text>
          <TextInput
            style={[styles.textArea, { color: colors.text, borderColor: colors.border }]}
            value={description}
            onChangeText={setDescription}
            placeholder={t('refund.description_placeholder')}
            placeholderTextColor={colors.textSecondary}
            multiline
            numberOfLines={4}
            textAlignVertical="top"
          />
        </View>

        {/* 提交按钮 */}
        <TouchableOpacity
          style={[styles.submitBtn, { backgroundColor: colors.primary }]}
          onPress={handleSubmit}
          disabled={submitting}
        >
          {submitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitBtnText}>{t('refund.submit')}</Text>
          )}
        </TouchableOpacity>

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {/* 退款历史弹窗 */}
      <Modal
        visible={showHistory}
        transparent
        animationType="slide"
        onRequestClose={() => setShowHistory(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity style={styles.modalBg} onPress={() => setShowHistory(false)} />
          <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.text }]}>{t('refund.history_title')}</Text>
              <TouchableOpacity onPress={() => setShowHistory(false)}>
                <IconSymbol name="xmark" size={24} color={colors.textSecondary} />
              </TouchableOpacity>
            </View>
            {refundHistory.length === 0 ? (
              <View style={styles.emptyHistory}>
                <Text style={[styles.emptyHistoryText, { color: colors.textSecondary }]}>
                  {t('refund.no_history')}
                </Text>
              </View>
            ) : (
              <FlatList
                data={refundHistory}
                keyExtractor={(item) => String(item.id)}
                renderItem={({ item }) => {
                  const statusInfo = REFUND_STATUS_MAP[item.status] || { text: '-', color: colors.textSecondary }
                  return (
                    <View style={[styles.historyItem, { borderBottomColor: colors.border }]}>
                      <View style={styles.historyItemHeader}>
                        <Text style={[styles.historyItemTime, { color: colors.textSecondary }]}>
                          {new Date(item.createTime).toLocaleString()}
                        </Text>
                        <Text style={[styles.historyItemStatus, { color: statusInfo.color }]}>
                          {statusInfo.text}
                        </Text>
                      </View>
                      <Text style={[styles.historyItemAmount, { color: colors.primary }]}>
                        ¥{item.refundAmount}
                      </Text>
                    </View>
                  )
                }}
              />
            )}
          </View>
        </View>
      </Modal>
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
  historyBtn: { padding: 8 },
  content: { flex: 1 },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  section: {
    padding: 16,
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  typeRow: {
    flexDirection: 'row',
    gap: 12,
  },
  typeBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
  },
  typeBtnText: {
    fontSize: 14,
  },
  reasonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  reasonBtn: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
  },
  reasonBtnText: {
    fontSize: 14,
  },
  amountRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  amountSymbol: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  amountInput: {
    fontSize: 28,
    fontWeight: 'bold',
    flex: 1,
    marginLeft: 4,
  },
  amountHint: {
    fontSize: 12,
    marginTop: 8,
  },
  textArea: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    minHeight: 100,
    fontSize: 14,
  },
  submitBtn: {
    marginHorizontal: 16,
    marginTop: 24,
    paddingVertical: 14,
    borderRadius: 24,
    alignItems: 'center',
  },
  submitBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomSpacer: { height: 40 },
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalBg: {
    ...StyleSheet.absoluteFillObject,
  },
  modalContent: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '60%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  emptyHistory: {
    padding: 40,
    alignItems: 'center',
  },
  emptyHistoryText: {
    fontSize: 14,
  },
  historyItem: {
    padding: 16,
    borderBottomWidth: 1,
  },
  historyItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  historyItemTime: {
    fontSize: 12,
  },
  historyItemStatus: {
    fontSize: 14,
    fontWeight: '500',
  },
  historyItemAmount: {
    fontSize: 16,
    fontWeight: 'bold',
  },
})
