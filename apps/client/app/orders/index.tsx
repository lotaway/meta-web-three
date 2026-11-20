import React, { useState, useCallback, useRef } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  Image,
  Alert,
} from 'react-native'
import { useRouter, useLocalSearchParams } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTranslation } from 'react-i18next'
import { useFocusEffect } from '@react-navigation/native'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { orderApi } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'

type OrderStatus = 'all' | 'unpaid' | 'undelivered' | 'received' | 'completed' | 'refund'
type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light
type AppRouter = ReturnType<typeof useRouter>
type UserId = NonNullable<ReturnType<typeof useAuth>['userId']>
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>
type SetOrders = React.Dispatch<React.SetStateAction<any[]>>
type PageNumRef = React.MutableRefObject<number>

const STATUS_TABS: { key: OrderStatus; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'unpaid', label: '待付款' },
  { key: 'undelivered', label: '待发货' },
  { key: 'received', label: '待收货' },
  { key: 'completed', label: '已完成' },
  { key: 'refund', label: '退款' },
]

const STATUS_MAP: Record<OrderStatus, number | undefined> = {
  all: undefined,
  unpaid: 0,
  undelivered: 1,
  received: 2,
  completed: 3,
  refund: 4,
}

const PAGE_SIZE = 10

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

async function fetchOrderPage(
  userId: UserId,
  status: number | undefined,
  pageNum: number
): Promise<any[] | undefined> {
  const response = await orderApi.list4({
    userId,
    status,
    pageNum,
    pageSize: PAGE_SIZE,
  } as any)
  return response.data ? (response.data as any[]) : undefined
}

function beginOrdersLoad(
  isRefresh: boolean,
  hasMore: boolean,
  setLoading: SetBoolean,
  setLoadingMore: SetBoolean,
  pageNumRef: PageNumRef
): boolean {
  if (isRefresh) {
    setLoading(true)
    pageNumRef.current = 1
    return true
  }
  if (!hasMore) return false
  setLoadingMore(true)
  return true
}

function applyLoadedOrders(
  newOrders: any[],
  isRefresh: boolean,
  setOrders: SetOrders,
  setHasMore: SetBoolean,
  pageNumRef: PageNumRef
) {
  if (isRefresh) {
    setOrders(newOrders)
  } else {
    setOrders(prev => [...prev, ...newOrders])
  }
  setHasMore(newOrders.length >= PAGE_SIZE)
  pageNumRef.current += 1
}

interface OrdersLoadContext {
  userId: UserId | undefined
  status: number | undefined
  isRefresh: boolean
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  pageNumRef: PageNumRef
  setOrders: SetOrders
  setHasMore: SetBoolean
  setLoading: SetBoolean
  setLoadingMore: SetBoolean
}

async function runOrdersLoad(ctx: OrdersLoadContext): Promise<void> {
  if (!ctx.userId || ctx.loading || ctx.loadingMore) return
  if (!beginOrdersLoad(ctx.isRefresh, ctx.hasMore, ctx.setLoading, ctx.setLoadingMore, ctx.pageNumRef)) return
  try {
    const newOrders = await fetchOrderPage(ctx.userId, ctx.status, ctx.pageNumRef.current)
    if (newOrders) applyLoadedOrders(newOrders, ctx.isRefresh, ctx.setOrders, ctx.setHasMore, ctx.pageNumRef)
  } catch (error) {
    console.error('Failed to load orders:', error)
    Alert.alert('Error', 'Failed to load orders.')
  } finally {
    ctx.setLoading(false)
    ctx.setLoadingMore(false)
  }
}

function useListState(initialStatus: OrderStatus) {
  const [activeTab, setActiveTab] = useState<OrderStatus>(initialStatus)
  const [orders, setOrders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const pageNumRef = useRef(1)
  return { activeTab, setActiveTab, orders, setOrders, loading, setLoading, loadingMore, setLoadingMore, hasMore, setHasMore, pageNumRef }
}

type ListState = ReturnType<typeof useListState>

function buildLoadContext(userId: number | null, status: number | undefined, isRefresh: boolean, state: ListState): OrdersLoadContext {
  return {
    userId: userId ?? undefined, status, isRefresh,
    loading: state.loading, loadingMore: state.loadingMore, hasMore: state.hasMore,
    pageNumRef: state.pageNumRef, setOrders: state.setOrders, setHasMore: state.setHasMore,
    setLoading: state.setLoading, setLoadingMore: state.setLoadingMore,
  }
}

function createListActions(router: AppRouter, setActiveTab: (tab: OrderStatus) => void, loadOrders: (isRefresh?: boolean) => void) {
  return {
    onSelectTab: setActiveTab,
    onOrderPress: (orderId: number) => router.push({ pathname: '/orders/[id]', params: { id: orderId } }),
    onLoadMore: () => loadOrders(false),
    onGoShopping: () => router.replace('/(tabs)'),
  }
}

type ListActions = ReturnType<typeof createListActions>

function useOrdersController() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()
  const params = useLocalSearchParams()
  const state = useListState((params.status as OrderStatus) || 'all')
  const loadOrders = useCallback(
    (isRefresh = false) => runOrdersLoad(buildLoadContext(userId, STATUS_MAP[state.activeTab], isRefresh, state)),
    [userId, state.activeTab, state.loading, state.loadingMore, state.hasMore]
  )
  React.useEffect(() => { loadOrders(true) }, [state.activeTab])
  useFocusEffect(useCallback(() => { loadOrders() }, [loadOrders]))
  return { t, colors, router, ...state, ...createListActions(router, state.setActiveTab, loadOrders) }
}

interface OrdersViewProps extends ListActions {
  colors: AppColors
  t: Translate
  router: AppRouter
  activeTab: OrderStatus
  orders: any[]
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
}

function OrdersHeader({ colors, t, router }: Pick<OrdersViewProps, 'colors' | 't' | 'router'>) {
  return (
    <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
        <IconSymbol name="chevron.left" size={24} color={colors.text} />
      </TouchableOpacity>
      <Text style={[styles.headerTitle, { color: colors.text }]}>{t('orders.title')}</Text>
      <View style={styles.headerRight} />
    </View>
  )
}

function OrderTabBar({ activeTab, colors, t, onSelectTab }: Pick<OrdersViewProps, 'activeTab' | 'colors' | 't' | 'onSelectTab'>) {
  return (
    <View style={[styles.tabBar, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
      {STATUS_TABS.map(tab => (
        <TouchableOpacity
          key={tab.key}
          style={[styles.tab, activeTab === tab.key && styles.tabActive]}
          onPress={() => onSelectTab(tab.key)}
        >
          <Text style={[styles.tabText, { color: activeTab === tab.key ? colors.primary : colors.textSecondary }]}>
            {t(`orders.tab_${tab.key}`)}
          </Text>
          {activeTab === tab.key && <View style={[styles.tabIndicator, { backgroundColor: colors.primary }]} />}
        </TouchableOpacity>
      ))}
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

function EmptyState({ colors, t, onGoShopping }: Pick<OrdersViewProps, 'colors' | 't' | 'onGoShopping'>) {
  return (
    <View style={styles.emptyContainer}>
      <IconSymbol name="list.bullet.rectangle" size={60} color={colors.textSecondary} />
      <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('orders.no_orders')}</Text>
      <TouchableOpacity style={[styles.goShoppingBtn, { backgroundColor: colors.primary }]} onPress={onGoShopping}>
        <Text style={styles.goShoppingBtnText}>{t('orders.go_shopping')}</Text>
      </TouchableOpacity>
    </View>
  )
}

function BorderActionButton({ label, colors }: { label: string; colors: AppColors }) {
  return (
    <TouchableOpacity style={[styles.actionBtn, { borderColor: colors.border }]}>
      <Text style={[styles.actionBtnText, { color: colors.text }]}>{label}</Text>
    </TouchableOpacity>
  )
}

function PayActionButton({ label, colors }: { label: string; colors: AppColors }) {
  return (
    <TouchableOpacity style={[styles.actionBtn, styles.payBtn, { backgroundColor: colors.primary }]}>
      <Text style={styles.payBtnText}>{label}</Text>
    </TouchableOpacity>
  )
}

function ConfirmActionButton({ label, colors }: { label: string; colors: AppColors }) {
  return (
    <TouchableOpacity style={[styles.actionBtn, styles.confirmBtn, { backgroundColor: colors.primary }]}>
      <Text style={styles.confirmBtnText}>{label}</Text>
    </TouchableOpacity>
  )
}

function OrderFooterActions({ status, colors, t }: { status: number; colors: AppColors; t: Translate }) {
  return (
    <View style={styles.orderActions}>
      {status === 0 && (
        <>
          <BorderActionButton label={t('orders.cancel')} colors={colors} />
          <PayActionButton label={t('orders.pay')} colors={colors} />
        </>
      )}
      {status === 2 && <ConfirmActionButton label={t('orders.confirm_receive')} colors={colors} />}
      {(status === 1 || status === 2) && <BorderActionButton label={t('orders.refund')} colors={colors} />}
    </View>
  )
}

function OrderCardFooter({ item, colors, t }: { item: any; colors: AppColors; t: Translate }) {
  return (
    <View style={[styles.orderFooter, { borderTopColor: colors.border }]}>
      <Text style={[styles.totalText, { color: colors.text }]}>
        {t('orders.total')}: <Text style={[styles.totalPrice, { color: colors.primary }]}>¥{item.payAmount || item.totalAmount}</Text>
      </Text>
      <OrderFooterActions status={item.status} colors={colors} t={t} />
    </View>
  )
}

function OrderProductRow({ product, colors }: { product: any; colors: AppColors }) {
  return (
    <View style={styles.orderItemRow}>
      <Image source={{ uri: product.productPic || '' }} style={styles.productImage} />
      <View style={styles.productInfo}>
        <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
          {product.productName}
        </Text>
        <View style={styles.productFooter}>
          <Text style={[styles.productPrice, { color: colors.primary }]}>¥{product.productPrice}</Text>
          <Text style={[styles.productQty, { color: colors.textSecondary }]}>x{product.productQuantity}</Text>
        </View>
      </View>
    </View>
  )
}

function OrderProductList({ item, colors, t }: { item: any; colors: AppColors; t: Translate }) {
  return (
    <View style={styles.orderContent}>
      {item.orderItems?.slice(0, 3).map((product: any, index: number) => (
        <OrderProductRow key={index} product={product} colors={colors} />
      ))}
      {item.orderItems?.length > 3 && (
        <Text style={[styles.moreText, { color: colors.textSecondary }]}>
          {t('orders.more_items', { count: item.orderItems.length - 3 })}
        </Text>
      )}
    </View>
  )
}

function OrderCardHeader({ item, colors, t }: { item: any; colors: AppColors; t: Translate }) {
  return (
    <View style={styles.orderHeader}>
      <Text style={[styles.orderNo, { color: colors.text }]}>
        {t('orders.order_no')}: {item.orderSn || item.id}
      </Text>
      <Text style={[styles.statusText, { color: getStatusColor(item.status, colors) }]}>
        {getStatusText(item.status, t)}
      </Text>
    </View>
  )
}

function OrderCard({ item, colors, t, onPress }: { item: any; colors: AppColors; t: Translate; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.orderCard, { backgroundColor: colors.card, borderColor: colors.border }]}
      onPress={onPress}
    >
      <OrderCardHeader item={item} colors={colors} t={t} />
      <OrderProductList item={item} colors={colors} t={t} />
      <OrderCardFooter item={item} colors={colors} t={t} />
    </TouchableOpacity>
  )
}

function ListFooterMessage({ colors, t }: { colors: AppColors; t: Translate }) {
  return (
    <View style={styles.footer}>
      <Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text>
    </View>
  )
}

function ListFooter({ loadingMore, hasMore, ordersLength, colors, t }: {
  loadingMore: boolean
  hasMore: boolean
  ordersLength: number
  colors: AppColors
  t: Translate
}) {
  if (loadingMore) {
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="small" color={colors.primary} />
      </View>
    )
  }
  if (!hasMore && ordersLength > 0) return <ListFooterMessage colors={colors} t={t} />
  return null
}

function OrderList(props: OrdersViewProps) {
  const { orders, colors, t, onOrderPress, onLoadMore } = props
  return (
    <FlatList
      data={orders}
      keyExtractor={(item) => String(item.id)}
      renderItem={({ item }) => (
        <OrderCard item={item} colors={colors} t={t} onPress={() => onOrderPress(item.id)} />
      )}
      contentContainerStyle={styles.listContent}
      onEndReached={onLoadMore}
      onEndReachedThreshold={0.5}
      ListFooterComponent={<ListFooter {...props} ordersLength={orders.length} />}
    />
  )
}

function OrdersView(props: OrdersViewProps) {
  const { colors, orders, loading } = props
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <OrdersHeader {...props} />
      <OrderTabBar {...props} />
      {loading ? (
        <LoadingState colors={colors} />
      ) : orders.length === 0 ? (
        <EmptyState {...props} />
      ) : (
        <OrderList {...props} />
      )}
    </SafeAreaView>
  )
}

export default function OrdersScreen() {
  const controller = useOrdersController()
  return <OrdersView {...controller} />
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
  tabBar: {
    flexDirection: 'row',
    borderBottomWidth: 1,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    position: 'relative',
  },
  tabActive: {},
  tabText: { fontSize: 14 },
  tabIndicator: {
    position: 'absolute',
    bottom: 0,
    width: 20,
    height: 3,
    borderRadius: 2,
  },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    marginBottom: 24,
  },
  goShoppingBtn: {
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 24,
  },
  goShoppingBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  listContent: {
    padding: 12,
  },
  footer: {
    padding: 16,
    alignItems: 'center',
  },
  orderCard: {
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
    overflow: 'hidden',
  },
  orderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  orderNo: { fontSize: 14 },
  statusText: { fontSize: 14, fontWeight: '500' },
  orderContent: {
    padding: 12,
  },
  orderItemRow: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  productImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
  },
  productInfo: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between',
  },
  productName: {
    fontSize: 14,
    lineHeight: 20,
  },
  productFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  productPrice: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  productQty: {
    fontSize: 14,
  },
  moreText: {
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8,
  },
  orderFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderTopWidth: 1,
  },
  totalText: {
    fontSize: 14,
  },
  totalPrice: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  orderActions: {
    flexDirection: 'row',
    gap: 10,
  },
  actionBtn: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
  },
  actionBtnText: { fontSize: 13 },
  payBtn: {
    borderWidth: 0,
  },
  payBtnText: { color: '#fff', fontSize: 13 },
  confirmBtn: {
    borderWidth: 0,
  },
  confirmBtnText: { color: '#fff', fontSize: 13 },
})
