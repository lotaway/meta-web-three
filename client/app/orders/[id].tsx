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
import { orderApi } from '@/api/generated'
import { useAuth } from '@/contexts/AuthContext'
import { OrderInfo } from '@/components/order/OrderInfo'
import { ProductList } from '@/components/order/ProductList'
import { AddressInfo } from '@/components/order/AddressInfo'
import { PaymentInfo } from '@/components/order/PaymentInfo'
import { ActionButtons } from '@/components/order/ActionButtons'

export default function OrderDetailScreen() {
  const { t } = useTranslation()
  const router = useRouter()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const { userId } = useAuth()
  const params = useLocalSearchParams()
  const orderId = params.id ? parseInt(params.id as string, 10) : 0

  const [order, setOrder] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadOrder()
  }, [orderId])

  const loadOrder = async () => {
    setLoading(true)
    try {
      const response = await orderApi.detail({ id: orderId })
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

  const executeCancel = async () => {
    try {
      await orderApi.cancel({ id: orderId } as any)
      Alert.alert(t('common.success'), t('orders.cancel_success'))
      loadOrder()
    } catch (error) {
      Alert.alert(t('common.error'), t('orders.cancel_failed'))
    }
  }

  const handleCancel = () => {
    Alert.alert(
      t('orders.cancel_confirm_title'),
      t('orders.cancel_confirm_message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('orders.confirm'),
          style: 'destructive',
          onPress: executeCancel,
        },
      ]
    )
  }

  const executeConfirmReceive = async () => {
    try {
      await orderApi.confirmReceive({ id: orderId } as any)
      Alert.alert(t('common.success'), t('orders.confirm_receive_success'))
      loadOrder()
    } catch (error) {
      Alert.alert(t('common.error'), t('orders.confirm_receive_failed'))
    }
  }

  const handleConfirmReceive = () => {
    Alert.alert(
      t('orders.confirm_receive_title'),
      t('orders.confirm_receive_message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('orders.confirm'),
          onPress: executeConfirmReceive,
        },
      ]
    )
  }

  const handlePay = () => {
    router.push({ pathname: '/checkout', params: { orderId } })
  }

  const handleViewLogistics = () => {
    router.push({ pathname: '/orders/[id]/logistics', params: { id: orderId } })
  }

  const handleRefund = () => {
    router.push({ pathname: '/orders/[id]/refund', params: { id: orderId } })
  }

  const handleReview = (item: any) => {
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

  const handleContactDelivery = () => {
    if (order?.deliveryCompanyPhone) {
      Linking.openURL(`tel:${order.deliveryCompanyPhone}`)
    }
  }

  const handleReviewItem = () => {
    const item = order.orderItems?.[0]
    if (item) handleReview(item)
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

  if (!order) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
            <IconSymbol name="chevron.left" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>{t('orders.detail_title')}</Text>
        </View>
        <View style={styles.emptyContainer}>
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{t('orders.not_found')}</Text>
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
        <Text style={[styles.headerTitle, { color: colors.text }]}>{t('orders.detail_title')}</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content}>
        <View style={[styles.statusSection, { backgroundColor: colors.card }]}>
          <IconSymbol
            name={order.status === 0 ? 'creditcard' : order.status === 3 ? 'checkmark.seal.fill' : 'shippingbox'}
            size={40}
            color={getStatusColor(order.status)}
          />
          <Text style={[styles.statusText, { color: getStatusColor(order.status) }]}>
            {getStatusText(order.status)}
          </Text>
          {order.status === 2 && order.deliveryCompanyName && (
            <TouchableOpacity style={styles.logisticsBtn} onPress={handleViewLogistics}>
              <Text style={styles.logisticsBtnText}>
                {t('orders.view_logistics')} ({order.deliveryCompanyName})
              </Text>
              <IconSymbol name="chevron.right" size={16} color="#fff" />
            </TouchableOpacity>
          )}
        </View>

        {order.receiverName && (
          <AddressInfo
            receiverName={order.receiverName}
            receiverPhone={order.receiverPhone}
            receiverProvince={order.receiverProvince}
            receiverCity={order.receiverCity}
            receiverRegion={order.receiverRegion}
            receiverDetailAddress={order.receiverDetailAddress}
          />
        )}

        <ProductList
          orderItems={order.orderItems}
          status={order.status}
          onReview={handleReview}
        />

        <OrderInfo order={order} />

        <PaymentInfo
          totalAmount={order.totalAmount}
          freightAmount={order.freightAmount}
          couponAmount={order.couponAmount}
          payAmount={order.payAmount}
        />

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {order.status !== 4 && (
        <ActionButtons
          status={order.status}
          onPay={handlePay}
          onCancel={handleCancel}
          onConfirmReceive={handleConfirmReceive}
          onRefund={handleRefund}
          onContactDelivery={handleContactDelivery}
          onReview={handleReviewItem}
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
