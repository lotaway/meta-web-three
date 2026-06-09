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
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import * as ImagePicker from 'expo-image-picker'
import { RefundTypeSelector } from '@/components/refund/RefundTypeSelector'
import { ReasonSelector } from '@/components/refund/ReasonSelector'
import { ProofImageUploader } from '@/components/refund/ProofImageUploader'
import { RefundHistoryList } from '@/components/refund/RefundHistoryList'
import { orderApi, API_BASE_URL, DEFAULT_USER_ID } from '@/api/generated'

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
  const [proofImages, setProofImages] = useState<string[]>([])

  useEffect(() => {
    loadOrder()
    loadRefundHistory()
  }, [orderId])

  const loadOrder = async () => {
    if (!orderId) return
    setLoading(true)
    try {
      const response = await orderApi.detail({
        xUserId: DEFAULT_USER_ID,
        id: Number(orderId),
      })
      if (response.data) {
        setOrder(response.data)
        setRefundAmount(String(response.data.order?.orderAmount || response.data.totalPrice || 0))
      }
    } catch (error) {
      console.error('Failed to load order:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadRefundHistory = async () => {
    if (!orderId) return
    try {
      const response = await fetch(`${API_BASE_URL}/order-service/returnApply/order/${orderId}`)
      const result = await response.json()
      if (result.data) {
        setRefundHistory(result.data)
      }
    } catch (error) {
      console.error('Failed to load refund history:', error)
    }
  }

  const handlePickImage = async () => {
    if (proofImages.length >= 5) {
      Alert.alert(t('common.error'), '最多只能上传5张图片')
      return
    }
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (!permissionResult.granted) {
      Alert.alert(t('common.error'), '需要相册权限才能选择图片')
      return
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 0.8,
    })
    if (!result.canceled && result.assets.length > 0) {
      setProofImages(prev => [...prev, result.assets[0].uri])
    }
  }

  const handleRemoveImage = (index: number) => {
    setProofImages(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if (!selectedReason) return Alert.alert(t('common.error'), t('refund.reason_required'))
    if (!refundAmount || parseFloat(refundAmount) <= 0) return Alert.alert(t('common.error'), t('refund.amount_required'))
    setSubmitting(true)
    try {
      const result = await callRefundApi()
      if (result.code === 200) return showSuccess()
    } catch (error) {
      Alert.alert(t('common.error'), t('refund.submit_failed'))
    } finally {
      setSubmitting(false)
    }
  }

  const callRefundApi = async () => {
    const response = await fetch(
      `${API_BASE_URL}/order-service/order/${orderId}/refund`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-user-id': String(DEFAULT_USER_ID),
        },
        body: JSON.stringify({
          orderId: Number(orderId),
          reason: getReasonText(selectedReason),
          returnAmount: parseFloat(refundAmount),
          description,
          proofPics: proofImages.join(','),
          refundType,
        }),
      }
    )
    return await response.json()
  }

  const getReasonText = (reasonId: number | null) => {
    const reasons = ['质量问题', '商品不符', '发货太慢', '七天无理由', '其他']
    return reasonId != null ? reasons[reasonId] || '其他' : '其他'
  }

  const showSuccess = () => {
    Alert.alert(t('common.success'), t('refund.submit_success'), [
      { text: 'OK', onPress: () => router.back() },
    ])
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
        <RefundTypeSelector refundType={refundType} setRefundType={setRefundType} />

        <ReasonSelector selectedReason={selectedReason} setSelectedReason={setSelectedReason} />

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

        <ProofImageUploader
          proofImages={proofImages}
          onRemoveImage={handleRemoveImage}
          onPickImage={handlePickImage}
        />

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

      <RefundHistoryList
        visible={showHistory}
        onClose={() => setShowHistory(false)}
        refundHistory={refundHistory}
      />
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
})
