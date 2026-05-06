import React, { useState, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  Image,
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

export default function OrdersScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()
  const params = useLocalSearchParams()
  
  const initialStatus = (params.status as OrderStatus) || 'all'
  const [activeTab, setActiveTab] = useState<OrderStatus>(initialStatus)
  const [orders, setOrders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const pageNumRef = useRef(1)
  const PAGE_SIZE = 10

  const loadOrders = useCallback(async (isRefresh = false) => {
    if (!userId || loading || loadingMore) return

    if (isRefresh) {
      setLoading(true)
      pageNumRef.current = 1
    } else {
      if (!hasMore) return
      setLoadingMore(true)
    }

    try {
      const response = await orderApi.listByUserId({
        userId,
        status: STATUS_MAP[activeTab],
        pageNum: pageNumRef.current,
        pageSize: PAGE_SIZE,
      } as any)
      if (response.data) {
        const newOrders = response.data as any[]
        if (isRefresh) {
          setOrders(newOrders)
        } else {
          setOrders(prev => [...prev, ...newOrders])
        }
        setHasMore(newOrders.length >= PAGE_SIZE)
        pageNumRef.current += 1
      }
    } catch (error) {
      console.error('Failed to load orders:', error)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }, [userId, activeTab, loading, loadingMore, hasMore])

  React.useEffect(() => {
    loadOrders(true)
  }, [activeTab])

  useFocusEffect(
    useCallback(() => {
      loadOrders()
    }, [loadOrders])
  )

  const handleOrderPress = (orderId: number) => {
    router.push({ pathname: '/orders/[id]', params: { id: orderId } })
  }

  const getStatusText = (status: number): string => {
    const statusTexts: Record<number, string> = {
      0: t('orders.status_unpaid'),
      1: t('orders.status_undelivered'),
      2: t('orders.status_received'),
      3: t('orders.status_completed'),
      4: t('orders.status_refund'),
    }
    return statusTexts[status] || ''
  }

  const getStatusColor = (status: number): string => {
    const statusColors: Record<number, string> = {
      0: '#FF6B35',
      1: '#4A90E2',
      2: '#5fcda2',
      3: '#8E8E93',
      4: '#FF3B30',
    }
    return statusColors[status] || colors.textSecondary
  }

  const renderOrderItem = ({ item }: { item: any }) => (
    <TouchableOpacity
      style={[styles.orderCard, { backgroundColor: colors.card, borderColor: colors.border }]}
      onPress={() => handleOrderPress(item.id)}
    >
      <View style={styles.orderHeader}>
        <Text style={[styles.orderNo, { color: colors.text }]}>
          {t('orders.order_no')}: {item.orderSn || item.id}
        </Text>
        <Text style={[styles.statusText, { color: getStatusColor(item.status) }]}>
          {getStatusText(item.status)}
        </Text>
      </View>

      <View style={styles.orderContent}>
        {item.orderItems?.slice(0, 3).map((orderItem: any, index: number) => (
          <View key={index} style={styles.orderItemRow}>
            <Image
              source={{ uri: orderItem.productPic || '' }}
              style={styles.productImage}
            />
            <View style={styles.productInfo}>
              <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
                {orderItem.productName}
              </Text>
              <View style={styles.productFooter}>
                <Text style={[styles.productPrice, { color: colors.primary }]}>
                  ¥{orderItem.productPrice}
                </Text>
                <Text style={[styles.productQty, { color: colors.textSecondary }]}>
                  x{orderItem.productQuantity}
                </Text>
              </View>
            </View>
          </View>
        ))}
        {item.orderItems?.length > 3 && (
          <Text style={[styles.moreText, { color: colors.textSecondary }]}>
            {t('orders.more_items', { count: item.orderItems.length - 3 })}
          </Text>
        )}
      </View>

      <View style={[styles.orderFooter, { borderTopColor: colors.border }]}>
        <Text style={[styles.totalText, { color: colors.text }]}>
          {t('orders.total')}: <Text style={[styles.totalPrice, { color: colors.primary }]}>¥{item.payAmount || item.totalAmount}</Text>
        </Text>
        <View style={styles.orderActions}>
          {item.status === 0 && (
            <>
              <TouchableOpacity style={[styles.actionBtn, { borderColor: colors.border }]}>
                <Text style={[styles.actionBtnText, { color: colors.text }]}>{t('orders.cancel')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.actionBtn, styles.payBtn, { backgroundColor: colors.primary }]}>
                <Text style={styles.payBtnText}>{t('orders.pay')}</Text>
              </TouchableOpacity>
            </>
          )}
          {item.status === 2 && (
            <TouchableOpacity style={[styles.actionBtn, styles.confirmBtn, { backgroundColor: colors.primary }]}>
              <Text style={styles.confirmBtnText}>{t('orders.confirm_receive')}</Text>
            </TouchableOpacity>
          )}
          {(item.status === 1 || item.status === 2) && (
            <TouchableOpacity style={[styles.actionBtn, { borderColor: colors.border }]}>
              <Text style={[styles.actionBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </TouchableOpacity>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <IconSymbol name="chevron.left" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('orders.title')}</Text>
        <View style={styles.headerRight} />
      </View>

      <View style={[styles.tabBar, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        {STATUS_TABS.map(tab => (
          <TouchableOpacity
            key={tab.key}
            style={[
              styles.tab,
              activeTab === tab.key && styles.tabActive,
            ]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text
              style={[
                styles.tabText,
                { color: activeTab === tab.key ? colors.primary : colors.textSecondary },
              ]}
            >
              {t(`orders.tab_${tab.key}`)}
            </Text>
            {activeTab === tab.key && (
              <View style={[styles.tabIndicator, { backgroundColor: colors.primary }]} />
            )}
          </TouchableOpacity>
        ))}
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : orders.length === 0 ? (
        <View style={styles.emptyContainer}>
          <IconSymbol name="list.bullet.rectangle" size={60} color={colors.textSecondary} />
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('orders.no_orders')}</Text>
          <TouchableOpacity style={[styles.goShoppingBtn, { backgroundColor: colors.primary }]} onPress={() => router.replace('/(tabs)')}>
            <Text style={styles.goShoppingBtnText}>{t('orders.go_shopping')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={orders}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderOrderItem}
          contentContainerStyle={styles.listContent}
          onEndReached={() => loadOrders(false)}
          onEndReachedThreshold={0.5}
          ListFooterComponent={
            loadingMore ? (
              <View style={styles.footer}>
                <ActivityIndicator size="small" color={colors.primary} />
              </View>
            ) : !hasMore && orders.length > 0 ? (
              <View style={styles.footer}>
                <Text style={{ color: colors.textSecondary }}>{t('common.no_more_data')}</Text>
              </View>
            ) : null
          }
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
