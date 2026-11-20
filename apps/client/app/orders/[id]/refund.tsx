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

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type OrderId = ReturnType<typeof useLocalSearchParams>['id']
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetProofImages = React.Dispatch<React.SetStateAction<string[]>>
type SetOrder = React.Dispatch<React.SetStateAction<any>>
type SetAmount = React.Dispatch<React.SetStateAction<string>>
type SetHistory = React.Dispatch<React.SetStateAction<any[]>>

function fetchRefundOrder(orderId: OrderId) {
  return orderApi.detail1({
    xUserId: DEFAULT_USER_ID,
    id: Number(orderId),
  })
}

async function loadOrderData(orderId: OrderId, setOrder: SetOrder, setRefundAmount: SetAmount, setLoading: SetBoolean, t: Translate): Promise<void> {
  if (!orderId) return
  setLoading(true)
  try {
    const response = await fetchRefundOrder(orderId)
    if (response.data) {
      setOrder(response.data)
      setRefundAmount(String(response.data.order?.orderAmount || response.data.totalPrice || 0))
    }
  } catch (error) {
    console.error('Failed to load order:', error)
    Alert.alert('Error', 'Failed to load the order.')
  } finally {
    setLoading(false)
  }
}

async function loadHistoryData(orderId: OrderId, setRefundHistory: SetHistory): Promise<void> {
  if (!orderId) return
  try {
    const response = await fetch(`${API_BASE_URL}/mall-trade-service/returnApply/order/${orderId}`)
    const result = await response.json()
    if (result.data) {
      setRefundHistory(result.data)
    }
  } catch (error) {
    console.error('Failed to load refund history:', error)
    Alert.alert('Error', 'Failed to load the refund history.')
  }
}

async function requestPhotoPermission(t: Translate): Promise<boolean> {
  const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync()
  if (!permissionResult.granted) {
    Alert.alert(t('common.error'), '需要相册权限才能选择图片')
    return false
  }
  return true
}

async function pickProofImage(
  proofImages: string[],
  setProofImages: SetProofImages,
  t: Translate
): Promise<void> {
  if (proofImages.length >= 5) {
    Alert.alert(t('common.error'), '最多只能上传5张图片')
    return
  }
  if (!(await requestPhotoPermission(t))) return
  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    allowsEditing: false,
    quality: 0.8,
  })
  if (!result.canceled && result.assets.length > 0) {
    setProofImages(prev => [...prev, result.assets[0].uri])
  }
}

function getReasonText(reasonId: number | null) {
  const reasons = ['质量问题', '商品不符', '发货太慢', '七天无理由', '其他']
  return reasonId != null ? reasons[reasonId] || '其他' : '其他'
}

function buildRefundPayload(
  orderId: OrderId,
  reason: string,
  refundAmount: string,
  description: string,
  proofImages: string[],
  refundType: number
) {
  return {
    orderId: Number(orderId),
    reason,
    returnAmount: parseFloat(refundAmount),
    description,
    proofPics: proofImages.join(','),
    refundType,
  }
}

async function callRefundApi(orderId: OrderId, payload: ReturnType<typeof buildRefundPayload>): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/mall-trade-service/order/${orderId}/refund`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-user-id': String(DEFAULT_USER_ID),
      },
      body: JSON.stringify(payload),
    }
  )
  return await response.json()
}

function showSubmitSuccess(router: AppRouter, t: Translate) {
  Alert.alert(t('common.success'), t('refund.submit_success'), [
    { text: 'OK', onPress: () => router.back() },
  ])
}

interface RefundSubmitContext {
  t: Translate
  router: AppRouter
  orderId: OrderId
  selectedReason: number | null
  refundAmount: string
  description: string
  proofImages: string[]
  refundType: number
  setSubmitting: SetBoolean
}

function isRefundInputValid(ctx: RefundSubmitContext): boolean {
  if (!ctx.selectedReason) {
    Alert.alert(ctx.t('common.error'), ctx.t('refund.reason_required'))
    return false
  }
  if (!ctx.refundAmount || parseFloat(ctx.refundAmount) <= 0) {
    Alert.alert(ctx.t('common.error'), ctx.t('refund.amount_required'))
    return false
  }
  return true
}

async function submitRefund(ctx: RefundSubmitContext): Promise<void> {
  if (!isRefundInputValid(ctx)) return
  ctx.setSubmitting(true)
  try {
    const payload = buildRefundPayload(
      ctx.orderId,
      getReasonText(ctx.selectedReason),
      ctx.refundAmount,
      ctx.description,
      ctx.proofImages,
      ctx.refundType
    )
    const result = await callRefundApi(ctx.orderId, payload)
    if (result.code === 200) return showSubmitSuccess(ctx.router, ctx.t)
  } catch (error) {
    Alert.alert(ctx.t('common.error'), ctx.t('refund.submit_failed'))
  } finally {
    ctx.setSubmitting(false)
  }
}

function useRefundState() {
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
  return {
    order, setOrder, loading, setLoading, submitting, setSubmitting,
    refundType, setRefundType, refundAmount, setRefundAmount,
    selectedReason, setSelectedReason, description, setDescription,
    showHistory, setShowHistory, refundHistory, setRefundHistory,
    proofImages, setProofImages,
  }
}

type RefundState = ReturnType<typeof useRefundState>

function buildSubmitContext(t: Translate, router: AppRouter, orderId: OrderId, state: RefundState): RefundSubmitContext {
  return {
    t, router, orderId,
    selectedReason: state.selectedReason, refundAmount: state.refundAmount,
    description: state.description, proofImages: state.proofImages,
    refundType: state.refundType, setSubmitting: state.setSubmitting,
  }
}

function createRefundActions(t: Translate, router: AppRouter, orderId: OrderId, state: RefundState) {
  return {
    onBack: () => router.back(),
    onOpenHistory: () => state.setShowHistory(true),
    onCloseHistory: () => state.setShowHistory(false),
    onSetRefundType: state.setRefundType,
    onSetReason: state.setSelectedReason,
    onSetRefundAmount: state.setRefundAmount,
    onSetDescription: state.setDescription,
    onPickImage: () => pickProofImage(state.proofImages, state.setProofImages, t),
    onRemoveImage: (index: number) => state.setProofImages(prev => prev.filter((_, i) => i !== index)),
    onSubmit: () => submitRefund(buildSubmitContext(t, router, orderId, state)),
  }
}

type RefundActions = ReturnType<typeof createRefundActions>

function useRefundController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const params = useLocalSearchParams()
  const orderId = params.id
  const state = useRefundState()
  const loadOrder = () => loadOrderData(orderId, state.setOrder, state.setRefundAmount, state.setLoading, t)
  const loadRefundHistory = () => loadHistoryData(orderId, state.setRefundHistory)
  useEffect(() => { loadOrder(); loadRefundHistory() }, [orderId])
  return { t, colors, ...state, ...createRefundActions(t, router, orderId, state) }
}

interface RefundViewProps {
  colors: AppColors
  t: Translate
  order: any
  loading: boolean
  submitting: boolean
  refundType: number
  refundAmount: string
  selectedReason: number | null
  description: string
  proofImages: string[]
  showHistory: boolean
  refundHistory: any[]
  onBack: () => void
  onOpenHistory: () => void
  onCloseHistory: () => void
  onSetRefundType: (type: number) => void
  onSetReason: (reason: number) => void
  onSetRefundAmount: (value: string) => void
  onSetDescription: (value: string) => void
  onPickImage: () => void
  onRemoveImage: (index: number) => void
  onSubmit: () => void
}

function LoadingView({ colors }: { colors: AppColors }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    </SafeAreaView>
  )
}

function RefundHeader({ colors, t, onBack, onOpenHistory }: Pick<RefundViewProps, 'colors' | 't' | 'onBack' | 'onOpenHistory'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{t('refund.title')}</Text>
      <TouchableOpacity style={styles.historyBtn} onPress={onOpenHistory}>
        <IconSymbol name="clock.fill" size={20} color={colors.primary} />
      </TouchableOpacity>
    </View>
  )
}

function AmountInputRow({ colors, refundAmount, onSetRefundAmount }: Pick<RefundViewProps, 'colors' | 'refundAmount' | 'onSetRefundAmount'>) {
  return (
    <View style={styles.amountRow}>
      <Text style={[styles.amountSymbol, { color: colors.primary }]}>¥</Text>
      <TextInput
        style={[styles.amountInput, { color: colors.primary }]}
        value={refundAmount}
        onChangeText={onSetRefundAmount}
        keyboardType="decimal-pad"
        placeholder="0.00"
        placeholderTextColor={colors.textSecondary}
      />
    </View>
  )
}

function AmountSection({ colors, t, order, refundAmount, onSetRefundAmount }: Pick<RefundViewProps, 'colors' | 't' | 'order' | 'refundAmount' | 'onSetRefundAmount'>) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.amount_title')}</Text>
      <AmountInputRow colors={colors} refundAmount={refundAmount} onSetRefundAmount={onSetRefundAmount} />
      <Text style={[styles.amountHint, { color: colors.textSecondary }]}>
        {t('refund.amount_hint')}: ¥{order?.payAmount}
      </Text>
    </View>
  )
}

function DescriptionSection({ colors, t, description, onSetDescription }: Pick<RefundViewProps, 'colors' | 't' | 'description' | 'onSetDescription'>) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.description_title')}</Text>
      <TextInput
        style={[styles.textArea, { color: colors.text, borderColor: colors.border }]}
        value={description}
        onChangeText={onSetDescription}
        placeholder={t('refund.description_placeholder')}
        placeholderTextColor={colors.textSecondary}
        multiline
        numberOfLines={4}
        textAlignVertical="top"
      />
    </View>
  )
}

function SubmitButton({ colors, t, submitting, onSubmit }: Pick<RefundViewProps, 'colors' | 't' | 'submitting' | 'onSubmit'>) {
  return (
    <TouchableOpacity
      style={[styles.submitBtn, { backgroundColor: colors.primary }]}
      onPress={onSubmit}
      disabled={submitting}
    >
      {submitting ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <Text style={styles.submitBtnText}>{t('refund.submit')}</Text>
      )}
    </TouchableOpacity>
  )
}

function RefundForm(props: RefundViewProps) {
  return (
    <ScrollView style={styles.content}>
      <RefundTypeSelector refundType={props.refundType} setRefundType={props.onSetRefundType} />
      <ReasonSelector selectedReason={props.selectedReason} setSelectedReason={props.onSetReason} />
      <AmountSection {...props} />
      <DescriptionSection {...props} />
      <ProofImageUploader
        proofImages={props.proofImages}
        onRemoveImage={props.onRemoveImage}
        onPickImage={props.onPickImage}
      />
      <SubmitButton {...props} />
      <View style={styles.bottomSpacer} />
    </ScrollView>
  )
}

function RefundView(props: RefundViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <RefundHeader {...props} />
      <RefundForm {...props} />
      <RefundHistoryList
        visible={props.showHistory}
        onClose={props.onCloseHistory}
        refundHistory={props.refundHistory}
      />
    </SafeAreaView>
  )
}

export default function RefundScreen() {
  const controller = useRefundController()
  if (controller.loading) {
    return <LoadingView colors={controller.colors} />
  }
  return <RefundView {...controller} />
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
