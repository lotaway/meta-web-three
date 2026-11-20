import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Linking,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { orderApi, DEFAULT_USER_ID, API_BASE_URL } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'
import { OrderInfo } from '@/components/order/OrderInfo'
import { ProductList } from '@/components/order/ProductList'
import { AddressInfo } from '@/components/order/AddressInfo'
import { PaymentInfo } from '@/components/order/PaymentInfo'
import { ActionButtons } from '@/components/order/ActionButtons'

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>

function getStatusText(status: number, t: Translate): string {
  const statusTexts: Record<number, string> = {
    0: t('orders.status_unpaid'),
    1: t('orders.status_undelivered'),
    2: t('orders.status_received'),
    3: t('orders.status_completed'),
    4: t('orders.status_refund'),
  }
  return statusTexts[status] || ''
}

function getStatusColor(status: number, colors: AppColors): string {
  const statusColors: Record<number, string> = {
    0: '#FF6B35',
    1: '#4A90E2',
    2: '#5fcda2',
    3: '#8E8E93',
    4: '#FF3B30',
  }
  return statusColors[status] || colors.textSecondary
}

async function loadOrderDetail(
  orderId: number,
  setOrder: React.Dispatch<React.SetStateAction<any>>,
  setLoading: React.Dispatch<React.SetStateAction<boolean>>,
  t: Translate
): Promise<void> {
  setLoading(true)
  try {
    const response = await orderApi.detail1({ xUserId: DEFAULT_USER_ID, id: orderId } as any)
    if (response.data) {
      setOrder(response.data)
    }
  } catch (error) {
    console.error('Failed to load order:', error)
    Alert.alert(t('common.error'), t('orders.load_failed'))
  } finally {
    setLoading(false)
  }
}

async function cancelOrder(orderId: number, reload: () => void, t: Translate): Promise<void> {
  try {
    await orderApi.cancel({ id: orderId } as any)
    Alert.alert(t('common.success'), t('orders.cancel_success'))
    reload()
  } catch (error) {
    Alert.alert(t('common.error'), t('orders.cancel_failed'))
  }
}

async function receiveOrder(orderId: number, reload: () => void, t: Translate): Promise<void> {
  try {
    await orderApi.confirmReceive({ id: orderId } as any)
    Alert.alert(t('common.success'), t('orders.confirm_receive_success'))
    reload()
  } catch (error) {
    Alert.alert(t('common.error'), t('orders.confirm_receive_failed'))
  }
}

function contactDelivery(order: any) {
  if (order?.deliveryCompanyPhone) {
    Linking.openURL(`tel:${order.deliveryCompanyPhone}`)
  }
}

async function createOrderConversation(orderId: number, router: AppRouter): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/cs/conversation/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-Id': String(DEFAULT_USER_ID) },
      body: JSON.stringify({ channel: 'ORDER', orderId: Number(orderId) }),
    })
    const json = await res.json()
    if (json.data?.sessionId) router.push(`/cs/${json.data.sessionId}`)
  } catch {
    Alert.alert('提示', '创建会话失败')
  }
}

function showCancelConfirm(orderId: number, reload: () => void, t: Translate) {
  Alert.alert(
    t('orders.cancel_confirm_title'),
    t('orders.cancel_confirm_message'),
    [
      { text: t('common.cancel'), style: 'cancel' },
      { text: t('orders.confirm'), style: 'destructive', onPress: () => cancelOrder(orderId, reload, t) },
    ]
  )
}

function showReceiveConfirm(orderId: number, reload: () => void, t: Translate) {
  Alert.alert(
    t('orders.confirm_receive_title'),
    t('orders.confirm_receive_message'),
    [
      { text: t('common.cancel'), style: 'cancel' },
      { text: t('orders.confirm'), onPress: () => receiveOrder(orderId, reload, t) },
    ]
  )
}

function navigateToReview(orderId: number, router: AppRouter, item: any) {
  router.push({
    pathname: '/orders/[id]/review',
    params: {
      id: orderId,
      productId: item.productId,
      productName: item.productName,
      productPic: item.productPic,
    },
  })
}

function createDetailActions(router: AppRouter, t: Translate, orderId: number, order: any, reload: () => void) {
  return {
    onBack: () => router.back(),
    onPay: () => router.push({ pathname: '/checkout', params: { orderId } }),
    onViewLogistics: () => router.push({ pathname: '/orders/[id]/logistics', params: { id: orderId } }),
    onRefund: () => router.push({ pathname: '/orders/[id]/refund', params: { id: orderId } }),
    onReview: (item: any) => navigateToReview(orderId, router, item),
    onCancel: () => showCancelConfirm(orderId, reload, t),
    onConfirmReceive: () => showReceiveConfirm(orderId, reload, t),
    onContactDelivery: () => contactDelivery(order),
    onReviewItem: () => router.push(`/review?orderId=${String(orderId)}` as any),
    onContactService: () => createOrderConversation(orderId, router),
  }
}

type DetailActions = ReturnType<typeof createDetailActions>

function useDetailState() {
  const [order, setOrder] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  return { order, setOrder, loading, setLoading }
}

function useOrderDetailController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()
  const params = useLocalSearchParams()
  const orderId = params.id ? parseInt(params.id as string, 10) : 0
  const state = useDetailState()
  const loadOrder = () => loadOrderDetail(orderId, state.setOrder, state.setLoading, t)
  useEffect(() => { loadOrder() }, [orderId])
  const actions = createDetailActions(router, t, orderId, state.order, loadOrder)
  return { t, colors, ...state, ...actions }
}

interface DetailHeaderProps {
  title: string
  colors: AppColors
  onBack: () => void
  colored?: boolean
  showRight?: boolean
}

function DetailHeader({ title, colors, onBack, colored, showRight }: DetailHeaderProps) {
  const headerStyle = colored
    ? [styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]
    : styles.header
  return (
    <View style={headerStyle}>
      <TouchableOpacity style={styles.backBtn} onPress={onBack}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{title}</Text>
      {showRight && <View style={styles.headerRight} />}
    </View>
  )
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

function NotFoundView({ colors, t, onBack }: { colors: AppColors; t: Translate; onBack: () => void }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <DetailHeader title={t('orders.detail_title')} colors={colors} onBack={onBack} />
      <View style={styles.emptyContainer}>
        <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('orders.not_found')}</Text>
      </View>
    </SafeAreaView>
  )
}

interface StatusSectionProps {
  status: number
  deliveryCompanyName?: string
  colors: AppColors
  t: Translate
  onViewLogistics: () => void
}

function LogisticsButton({ deliveryCompanyName, t, onPress }: {
  deliveryCompanyName: string
  t: Translate
  onPress: () => void
}) {
  return (
    <TouchableOpacity style={styles.logisticsBtn} onPress={onPress}>
      <Text style={styles.logisticsBtnText}>
        {t('orders.view_logistics')} ({deliveryCompanyName})
      </Text>
      <IconSymbol name="chevron.right" size={16} color="#fff" />
    </TouchableOpacity>
  )
}

function StatusSection({ status, deliveryCompanyName, colors, t, onViewLogistics }: StatusSectionProps) {
  const statusColor = getStatusColor(status, colors)
  return (
    <View style={[styles.statusSection, { backgroundColor: colors.card }]}>
      <IconSymbol
        name={status === 0 ? 'creditcard' : status === 3 ? 'checkmark.seal.fill' : 'shippingbox'}
        size={40}
        color={statusColor}
      />
      <Text style={[styles.statusText, { color: statusColor }]}>
        {getStatusText(status, t)}
      </Text>
      {status === 2 && !!deliveryCompanyName && (
        <LogisticsButton deliveryCompanyName={deliveryCompanyName} t={t} onPress={onViewLogistics} />
      )}
    </View>
  )
}

function ReceiverAddress({ order }: { order: any }) {
  if (!order.receiverName) return null
  return (
    <AddressInfo
      receiverName={order.receiverName}
      receiverPhone={order.receiverPhone}
      receiverProvince={order.receiverProvince}
      receiverCity={order.receiverCity}
      receiverRegion={order.receiverRegion}
      receiverDetailAddress={order.receiverDetailAddress}
    />
  )
}

interface OrderDetailViewProps extends DetailActions {
  colors: AppColors
  t: Translate
  order: any
}

function OrderPayment({ order, colors }: { order: any; colors: AppColors }) {
  return (
    <PaymentInfo
      totalAmount={order.totalAmount}
      freightAmount={order.freightAmount}
      couponAmount={order.couponAmount}
      payAmount={order.payAmount}
    />
  )
}

function OrderSections(props: OrderDetailViewProps) {
  const { order, colors, t, onViewLogistics, onReview } = props
  return (
    <ScrollView style={styles.content}>
      <StatusSection status={order.status} deliveryCompanyName={order.deliveryCompanyName} colors={colors} t={t} onViewLogistics={onViewLogistics} />
      <ReceiverAddress order={order} />
      <ProductList orderItems={order.orderItems} status={order.status} onReview={onReview} />
      <OrderInfo order={order} />
      <OrderPayment order={order} colors={colors} />
      <View style={styles.bottomSpacer} />
    </ScrollView>
  )
}

function DetailActionBar(props: OrderDetailViewProps) {
  const { order, colors } = props
  if (order.status === 4) return null
  return (
    <ActionButtons
      status={order.status}
      orderId={order.id}
      onPay={props.onPay}
      onCancel={props.onCancel}
      onConfirmReceive={props.onConfirmReceive}
      onRefund={props.onRefund}
      onContactDelivery={props.onContactDelivery}
      onReview={props.onReviewItem}
      onContactService={props.onContactService}
    />
  )
}

function OrderDetailView(props: OrderDetailViewProps) {
  const { colors, t, onBack } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <DetailHeader title={t('orders.detail_title')} colors={colors} onBack={onBack} colored showRight />
      <OrderSections {...props} />
      <DetailActionBar {...props} />
    </SafeAreaView>
  )
}

export default function OrderDetailScreen() {
  const controller = useOrderDetailController()
  if (controller.loading) {
    return <LoadingView colors={controller.colors} />
  }
  if (!controller.order) {
    return <NotFoundView colors={controller.colors} t={controller.t} onBack={controller.onBack} />
  }
  return <OrderDetailView {...controller} order={controller.order} />
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
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyText: { fontSize: 14 },
  content: { flex: 1 },
  statusSection: {
    alignItems: 'center',
    paddingVertical: 30,
    marginBottom: 10,
  },
  statusText: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 12,
  },
  logisticsBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    backgroundColor: '#5fcda2',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 8,
  },
  logisticsBtnText: { color: '#fff', fontSize: 14 },
  bottomSpacer: { height: 80 },
})
