import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Image,
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

  const handlePay = async () => {
    router.push({ pathname: '/checkout', params: { orderId } })
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
          onPress: async () => {
            try {
              await orderApi.cancel({ id: orderId } as any)
              Alert.alert(t('common.success'), t('orders.cancel_success'))
              loadOrder()
            } catch (error) {
              Alert.alert(t('common.error'), t('orders.cancel_failed'))
            }
          },
        },
      ]
    )
  }

  const handleConfirmReceive = () => {
    Alert.alert(
      t('orders.confirm_receive_title'),
      t('orders.confirm_receive_message'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('orders.confirm'),
          onPress: async () => {
            try {
              await orderApi.confirmReceive({ id: orderId } as any)
              Alert.alert(t('common.success'), t('orders.confirm_receive_success'))
              loadOrder()
            } catch (error) {
              Alert.alert(t('common.error'), t('orders.confirm_receive_failed'))
            }
          },
        },
      ]
    )
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
        {/* 订单状态 */}
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

        {/* 收货地址 */}
        {order.receiverName && (
          <View style={[styles.section, { backgroundColor: colors.card }]}>
            <View style={styles.addressHeader}>
              <IconSymbol name="location.fill" size={18} color={colors.primary} />
              <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.address_title')}</Text>
            </View>
            <View style={styles.addressContent}>
              <Text style={[styles.addressName, { color: colors.text }]}>
                {order.receiverName} {order.receiverPhone}
              </Text>
              <Text style={[styles.addressDetail, { color: colors.textSecondary }]}>
                {order.receiverProvince}{order.receiverCity}{order.receiverRegion}{order.receiverDetailAddress}
              </Text>
            </View>
          </View>
        )}

        {/* 商品信息 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.products_title')}</Text>
          {order.orderItems?.map((item: any, index: number) => (
            <View key={index} style={[styles.productRow, { borderBottomColor: colors.border }]}>
              <TouchableOpacity
                style={styles.productRowContent}
                onPress={() => router.push({ pathname: '/product/[id]', params: { id: item.productId } })}
              >
                <Image source={{ uri: item.productPic || '' }} style={styles.productImage} />
                <View style={styles.productInfo}>
                  <Text numberOfLines={2} style={[styles.productName, { color: colors.text }]}>
                    {item.productName}
                  </Text>
                  <View style={styles.productFooter}>
                    <Text style={[styles.productPrice, { color: colors.primary }]}>¥{item.productPrice}</Text>
                    <Text style={[styles.productQty, { color: colors.textSecondary }]}>x{item.productQuantity}</Text>
                  </View>
                </View>
              </TouchableOpacity>
              {order.status === 3 && (
                <TouchableOpacity
                  style={styles.reviewItemBtn}
                  onPress={() => handleReview(item)}
                >
                  <IconSymbol name="pencil" size={16} color={colors.primary} />
                  <Text style={[styles.reviewItemText, { color: colors.primary }]}>评价</Text>
                </TouchableOpacity>
              )}
            </View>
          ))}
        </View>

        {/* 订单信息 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.info_title')}</Text>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.order_no')}</Text>
            <Text style={[styles.infoValue, { color: colors.text }]}>{order.orderSn || order.id}</Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.create_time')}</Text>
            <Text style={[styles.infoValue, { color: colors.text }]}>
              {order.createTime ? new Date(order.createTime).toLocaleString() : '-'}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.pay_time')}</Text>
            <Text style={[styles.infoValue, { color: colors.text }]}>
              {order.paymentTime ? new Date(order.paymentTime).toLocaleString() : '-'}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.pay_type')}</Text>
            <Text style={[styles.infoValue, { color: colors.text }]}>
              {order.payType === 0 ? t('orders.pay_type_wechat') :
               order.payType === 1 ? t('orders.pay_type_alipay') :
               order.payType === 2 ? t('orders.pay_type_stripe') : '-'}
            </Text>
          </View>
          {order.note && (
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.note')}</Text>
              <Text style={[styles.infoValue, { color: colors.text }]}>{order.note}</Text>
            </View>
          )}
        </View>

        {/* 费用明细 */}
        <View style={[styles.section, { backgroundColor: colors.card }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.amount_title')}</Text>
          <View style={styles.infoRow}>
            <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.product_amount')}</Text>
            <Text style={[styles.infoValue, { color: colors.text }]}>¥{order.totalAmount}</Text>
          </View>
          {order.freightAmount > 0 && (
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.freight')}</Text>
              <Text style={[styles.infoValue, { color: colors.text }]}>¥{order.freightAmount}</Text>
            </View>
          )}
          {order.couponAmount > 0 && (
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.coupon')}</Text>
              <Text style={[styles.infoValue, { color: '#FF3B30' }]}>-¥{order.couponAmount}</Text>
            </View>
          )}
          <View style={[styles.infoRow, styles.totalRow, { borderTopColor: colors.border }]}>
            <Text style={[styles.totalLabel, { color: colors.text }]}>{t('orders.pay_amount')}</Text>
            <Text style={[styles.totalValue, { color: colors.primary }]}>¥{order.payAmount}</Text>
          </View>
        </View>

        <View style={styles.bottomSpacer} />
      </ScrollView>

      {/* 底部操作栏 */}
      {order.status !== 4 && (
        <View style={[styles.bottomBar, { backgroundColor: colors.background, borderTopColor: colors.border }]}>
          {order.status === 0 && (
            <>
              <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={handleCancel}>
                <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.cancel')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.bottomBtn, styles.payBottomBtn, { backgroundColor: colors.primary }]} onPress={handlePay}>
                <Text style={styles.payBottomBtnText}>{t('orders.pay')}</Text>
              </TouchableOpacity>
            </>
          )}
          {order.status === 1 && (
            <>
              <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={handleRefund}>
                <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={handleContactDelivery}>
                <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.contact_delivery')}</Text>
              </TouchableOpacity>
            </>
          )}
          {order.status === 2 && (
            <>
              <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={handleRefund}>
                <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.bottomBtn, styles.confirmBottomBtn, { backgroundColor: colors.primary }]} onPress={handleConfirmReceive}>
                <Text style={styles.confirmBottomBtnText}>{t('orders.confirm_receive')}</Text>
              </TouchableOpacity>
            </>
          )}
          {order.status === 3 && (
            <>
              <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={() => router.push({ pathname: '/orders/[id]/refund', params: { id: orderId } })}>
                <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.bottomBtn, styles.confirmBottomBtn, { backgroundColor: colors.primary }]} onPress={() => {
                const item = order.orderItems?.[0]
                if (item) handleReview(item)
              }}>
                <Text style={styles.confirmBottomBtnText}>评价商品</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
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
  section: {
    marginBottom: 10,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  addressHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  addressContent: { paddingLeft: 26 },
  addressName: { fontSize: 16, fontWeight: '500' },
  addressDetail: { fontSize: 14, marginTop: 4, lineHeight: 20 },
  productRow: {
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  productRowContent: { flexDirection: 'row', flex: 1 },
  reviewItemBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#eee',
    backgroundColor: '#fff',
    alignSelf: 'center',
    gap: 4,
  },
  reviewItemText: {
    fontSize: 12,
    fontWeight: '500',
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
  productName: { fontSize: 14, lineHeight: 20 },
  productFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  productPrice: { fontSize: 16, fontWeight: 'bold' },
  productQty: { fontSize: 14 },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  infoLabel: { fontSize: 14 },
  infoValue: { fontSize: 14 },
  totalRow: {
    paddingTop: 12,
    marginTop: 4,
    borderTopWidth: 1,
  },
  totalLabel: { fontSize: 14, fontWeight: '600' },
  totalValue: { fontSize: 18, fontWeight: 'bold' },
  bottomSpacer: { height: 80 },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderTopWidth: 1,
    gap: 12,
  },
  bottomBtn: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
  },
  bottomBtnText: { fontSize: 14 },
  payBottomBtn: {
    borderWidth: 0,
  },
  payBottomBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  confirmBottomBtn: {
    borderWidth: 0,
  },
  confirmBottomBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' },
})
