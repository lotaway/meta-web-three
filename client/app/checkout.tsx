import React, { useState, useEffect, useCallback } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Modal,
  FlatList,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router, useLocalSearchParams, useFocusEffect } from 'expo-router'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { DEFAULT_USER_ID, orderApi, payApi, addressApi, couponApi } from '@/api/generated'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { pay, type PayResult } from '@/app/lib/payment'
import type { ApiResponseLong, ApiResponseMapStringObject, ApiResponseMapStringString, OrderItemCreate, MemberReceiveAddressDTO } from '@/src/generated/api/models'

type PayMethod = 'wechat' | 'alipay' | 'stripe'

interface PayMethodOption {
  id: PayMethod
  name: string
  icon: string
}

const PAY_METHODS: PayMethodOption[] = [
  { id: 'wechat', name: '微信支付', icon: 'weixin' },
  { id: 'alipay', name: '支付宝', icon: 'alipay' },
  { id: 'stripe', name: '银行卡', icon: 'creditcard' },
]

interface OrderInfo {
  orderId: number
  amount: number
  couponDiscount: number
  payAmount: number
  items: Array<{
    name: string
    price: number
    quantity: number
  }>
}

interface CheckoutItemParam {
  id: number
  productName: string
  productPic: string
  price: number
  quantity: number
}

interface AvailableCoupon {
  id: number
  name: string
  amount: number
  minPoint: number
  endTime: string
  useType: number
}

type PaymentParams = ApiResponseMapStringString['data']

function parseCheckoutItems(itemsParam: string | string[] | undefined): CheckoutItemParam[] {
  if (typeof itemsParam !== 'string' || itemsParam.length === 0) {
    return []
  }
  try {
    const parsed = JSON.parse(itemsParam)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function isSuccessCode(code?: string) {
  return code === '0000'
}

function mapItemsToOrderCreate(items: CheckoutItemParam[]): OrderItemCreate[] {
  return items.map((item) => ({
    productId: item.id,
    productName: item.productName,
    skuId: item.id,
    quantity: item.quantity,
    unitPrice: item.price,
    imageUrl: item.productPic,
  }))
}

function mapItemsForView(items: CheckoutItemParam[]): OrderInfo['items'] {
  return items.map((item) => ({
    name: item.productName,
    price: item.price,
    quantity: item.quantity,
  }))
}

async function createOrder(items: CheckoutItemParam[], addressId?: number, couponId?: number, remark?: string): Promise<ApiResponseLong> {
  return orderApi.create({
    xUserId: DEFAULT_USER_ID,
    orderCreateRequest: {
      items: mapItemsToOrderCreate(items),
      memberReceiveAddressId: addressId,
      couponId: couponId,
      remark: remark,
    },
  })
}

async function getPaymentParams(orderId: number, method: PayMethod): Promise<PaymentParams> {
  const request = {
    xUserId: DEFAULT_USER_ID,
    requestBody: { orderId },
  }
  const response = method === 'wechat'
    ? await payApi.getWechatParams(request)
    : method === 'alipay'
      ? await payApi.getAlipayParams(request)
      : await payApi.getStripeParams(request)
  if (!isSuccessCode(response.code) || !response.data) {
    throw new Error(response.message || '获取支付参数失败')
  }
  return response.data
}

async function verifyPayment(orderId: number, transactionId: string): Promise<ApiResponseMapStringObject> {
  return payApi.verifyPayment({
    xUserId: DEFAULT_USER_ID,
    requestBody: {
      orderId: String(orderId),
      transactionId,
    },
  })
}

async function markOrderPaid(orderId: number, method: PayMethod) {
  const payType = method === 'wechat' ? 1 : method === 'alipay' ? 2 : 3
  const response = await orderApi.paySuccess({ orderId, payType })
  if (!isSuccessCode(response.code)) {
    throw new Error(response.message || '订单支付状态更新失败')
  }
}

export default function CheckoutScreen() {
  const { items: itemsParam } = useLocalSearchParams<{ items?: string }>()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]
  const checkoutItems = parseCheckoutItems(itemsParam)

  const [orderInfo, setOrderInfo] = useState<OrderInfo | null>(null)
  const [selectedMethod, setSelectedMethod] = useState<PayMethod>('wechat')
  const [paying, setPaying] = useState(false)
  const [loading, setLoading] = useState(true)
  const [selectedAddress, setSelectedAddress] = useState<MemberReceiveAddressDTO | null>(null)
  const [couponModalVisible, setCouponModalVisible] = useState(false)
  const [availableCoupons, setAvailableCoupons] = useState<AvailableCoupon[]>([])
  const [selectedCoupon, setSelectedCoupon] = useState<AvailableCoupon | null>(null)
  const [loadingCoupons, setLoadingCoupons] = useState(false)
  const [orderRemark, setOrderRemark] = useState('')

  useEffect(() => {
    loadOrder()
  }, [])

  useFocusEffect(
    useCallback(() => {
      loadDefaultAddress()
    }, [])
  )

  const loadDefaultAddress = async () => {
    try {
      const response = await addressApi.list({ xUserId: DEFAULT_USER_ID })
      if (response.data && Array.isArray(response.data) && response.data.length > 0) {
        const defaultAddr = response.data.find((addr) => addr.defaultStatus === 1)
        setSelectedAddress(defaultAddr ?? response.data[0])
      }
    } catch (error) {
      console.error('Failed to load address:', error)
    }
  }

  const loadAvailableCoupons = async () => {
    setLoadingCoupons(true)
    try {
      const response = await couponApi.list({
        xUserId: DEFAULT_USER_ID,
        useStatus: 0,
      })
      if (response.data && Array.isArray(response.data)) {
        setAvailableCoupons(response.data as any)
      }
    } catch (error) {
      console.error('Failed to load coupons:', error)
    } finally {
      setLoadingCoupons(false)
    }
  }

  const handleSelectCoupon = (coupon: AvailableCoupon) => {
    setSelectedCoupon(coupon)
    setCouponModalVisible(false)
  }

  const handleClearCoupon = () => {
    setSelectedCoupon(null)
  }

  const loadOrder = async () => {
    try {
      if (checkoutItems.length === 0) {
        throw new Error('未找到可结算商品')
      }
      const orderResponse = await createOrder(checkoutItems, selectedAddress?.id)
      if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
        throw new Error(orderResponse.message || '创建订单失败')
      }
      const orderId = orderResponse.data
      const detailResponse = await orderApi.detail({
        xUserId: DEFAULT_USER_ID,
        id: orderId,
      })
      const detail = detailResponse.data
      setOrderInfo({
        orderId,
        amount: Number(detail?.order?.orderAmount ?? 0),
        couponDiscount: 0,
        payAmount: Number(detail?.order?.orderAmount ?? 0),
        items: detail?.items?.map((item) => ({
          name: item.productName ?? '',
          price: Number(item.unitPrice ?? 0),
          quantity: item.quantity ?? 0,
        })) ?? mapItemsForView(checkoutItems),
      })
    } catch (error) {
      Alert.alert('错误', '加载订单失败')
      router.back()
    } finally {
      setLoading(false)
    }
  }

  const handlePay = async () => {
    if (!orderInfo || paying) return
    setPaying(true)

    try {
      const finalAmount = orderInfo.amount - (selectedCoupon?.amount ?? 0)
      const payResult = await handlePaymentWithOrder(finalAmount)

      handlePayResult(payResult, orderInfo.orderId)
    } catch (error: any) {
      Alert.alert('支付异常', error.message || '请稍后重试')
    } finally {
      setPaying(false)
    }
  }

  const handlePaymentWithOrder = async (finalAmount: number): Promise<PayResult> => {
    let currentOrderId = orderInfo!.orderId

    if (selectedCoupon) {
      const orderResponse = await createOrder(
        checkoutItems,
        selectedAddress?.id,
        selectedCoupon.id,
        orderRemark
      )
      if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
        throw new Error(orderResponse.message || '创建订单失败')
      }
      currentOrderId = orderResponse.data
    }

    const params = await getPaymentParams(currentOrderId, selectedMethod)
    const result = await processPayment(selectedMethod, params)

    if (result.status === 'success') {
      if (!result.transactionId) {
        throw new Error('支付结果缺少交易凭证')
      }
      const verification = await verifyPayment(currentOrderId, result.transactionId)
      const valid = verification.data?.valid === true
      if (!valid) {
        throw new Error('支付校验失败')
      }
      await markOrderPaid(currentOrderId, selectedMethod)
    }

    return result
  }

  const processPayment = async (
    method: PayMethod,
    params: PaymentParams
  ): Promise<PayResult> => {
    switch (method) {
      case 'wechat':
        if (!params) throw new Error('微信支付参数获取失败')
        return await pay('wechat', params as any)

      case 'alipay':
        if (!params?.orderString) throw new Error('支付宝支付参数获取失败')
        return await pay('alipay', params.orderString)

      case 'stripe':
        if (!params?.clientSecret) throw new Error('银行卡支付参数获取失败')
        return await pay('stripe', {
          clientSecret: params.clientSecret,
          returnURL: params.returnURL || 'app://payment',
        })
    }
  }

  const handlePayResult = (result: PayResult, orderId: number) => {
    switch (result.status) {
      case 'success':
        Alert.alert('支付成功', `订单号: ${orderId}`, [
          { text: '确定', onPress: () => router.replace('/(tabs)/cart') },
        ])
        break
      case 'cancel':
        Alert.alert('取消支付', '您取消了支付')
        break
      case 'fail':
        Alert.alert('支付失败', result.message || '支付失败，请重试')
        break
    }
  }

  if (loading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.loading}>
          <Text style={{ color: colors.fontColorBase }}>加载中...</Text>
        </View>
      </SafeAreaView>
    )
  }

  if (!orderInfo) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.loading}>
          <Text style={{ color: colors.fontColorBase }}>订单不存在</Text>
        </View>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView style={styles.content}>
        <TouchableOpacity
          style={[styles.addressSection, { backgroundColor: colors.background }]}
          onPress={() => router.push('/address/list?source=checkout')}
        >
          {selectedAddress ? (
            <View style={styles.addressContent}>
              <IconSymbol name="location.fill" size={24} color={colors.primary} />
              <View style={styles.addressInfo}>
                <View style={styles.addressTop}>
                  <Text style={[styles.addressName, { color: colors.fontColorDark }]}>
                    {selectedAddress.name}
                  </Text>
                  <Text style={[styles.addressPhone, { color: colors.fontColorBase }]}>
                    {selectedAddress.phoneNumber}
                  </Text>
                </View>
                <Text style={[styles.addressDetail, { color: colors.fontColorLight }]}>
                  {selectedAddress.province} {selectedAddress.city} {selectedAddress.region} {selectedAddress.detailAddress}
                </Text>
              </View>
              <IconSymbol name="chevron.right" size={20} color={colors.fontColorDisabled} />
            </View>
          ) : (
            <View style={styles.addressContent}>
              <IconSymbol name="location" size={24} color={colors.fontColorDisabled} />
              <Text style={[styles.addressPlaceholder, { color: colors.fontColorDisabled }]}>
                请选择收货地址
              </Text>
              <IconSymbol name="chevron.right" size={20} color={colors.fontColorDisabled} />
            </View>
          )}
        </TouchableOpacity>

        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
            订单信息
          </Text>
          <View style={styles.orderInfo}>
            <Text style={[styles.orderId, { color: colors.fontColorBase }]}>
              订单号: {orderInfo.orderId}
            </Text>
            <Text style={[styles.orderItems, { color: colors.fontColorLight }]}>
              {orderInfo.items.map((item) => `${item.name} x${item.quantity}`).join(', ')}
            </Text>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.section, { backgroundColor: colors.background }]}
          onPress={() => {
            loadAvailableCoupons()
            setCouponModalVisible(true)
          }}
        >
          <View style={styles.couponRow}>
            <IconSymbol name="ticket" size={20} color={colors.primary} />
            <Text style={[styles.sectionTitle, { color: colors.fontColorDark, flex: 1 }]}>
              优惠券
            </Text>
            {selectedCoupon ? (
              <View style={styles.couponSelected}>
                <Text style={[styles.couponSelectedText, { color: colors.primary }]}>
                  -¥{selectedCoupon.amount}
                </Text>
                <TouchableOpacity onPress={handleClearCoupon} style={styles.clearCouponBtn}>
                  <IconSymbol name="xmark.circle.fill" size={18} color={colors.fontColorDisabled} />
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.couponPlaceholder}>
                <Text style={[styles.couponPlaceholderText, { color: colors.fontColorDisabled }]}>
                  选择优惠券
                </Text>
                <IconSymbol name="chevron.right" size={16} color={colors.fontColorDisabled} />
              </View>
            )}
          </View>
        </TouchableOpacity>

        <View style={[styles.section, { backgroundColor: colors.background }]}>
          <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
            支付方式
          </Text>
          {PAY_METHODS.map((method) => (
            <TouchableOpacity
              key={method.id}
              style={[
                styles.methodItem,
                selectedMethod === method.id && {
                  borderColor: colors.primary,
                  borderWidth: 2,
                },
              ]}
              onPress={() => setSelectedMethod(method.id)}
            >
              <View style={styles.methodLeft}>
                <IconSymbol
                  name={method.icon as any}
                  size={24}
                  color={
                    selectedMethod === method.id
                      ? colors.primary
                      : colors.fontColorBase
                  }
                />
                <Text
                  style={[
                    styles.methodName,
                    { color: colors.fontColorDark },
                    selectedMethod === method.id && { color: colors.primary },
                  ]}
                >
                  {method.name}
                </Text>
              </View>
              <IconSymbol
                name={
                  selectedMethod === method.id
                    ? 'checkmark.circle.fill'
                    : 'circle'
                }
                size={24}
                color={
                  selectedMethod === method.id
                    ? colors.primary
                    : colors.fontColorDisabled
                }
              />
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>

      <View style={[styles.footer, { backgroundColor: colors.background }]}>
        <View style={styles.footerTotal}>
          {selectedCoupon && (
            <Text style={[styles.footerDiscount, { color: colors.fontColorDisabled }]}>
              ¥{orderInfo.amount.toFixed(2)} - ¥{selectedCoupon.amount.toFixed(2)}
            </Text>
          )}
          <Text style={[styles.footerLabel, { color: colors.fontColorDark }]}>
            实付:
          </Text>
          <Text style={[styles.footerAmount, { color: colors.primary }]}>
            ¥{(orderInfo.amount - (selectedCoupon?.amount ?? 0)).toFixed(2)}
          </Text>
        </View>
        <TouchableOpacity
          style={[
            styles.payButton,
            { backgroundColor: paying ? colors.fontColorDisabled : colors.primary },
          ]}
          onPress={handlePay}
          disabled={paying}
        >
          <Text style={styles.payButtonText}>
            {paying ? '支付中...' : `立即支付`}
          </Text>
        </TouchableOpacity>
      </View>

      {/* 优惠券选择弹窗 */}
      <Modal
        visible={couponModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setCouponModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.background }]}>
            <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
              <Text style={[styles.modalTitle, { color: colors.fontColorDark }]}>
                选择优惠券
              </Text>
              <TouchableOpacity onPress={() => setCouponModalVisible(false)}>
                <IconSymbol name="xmark" size={24} color={colors.fontColorBase} />
              </TouchableOpacity>
            </View>
            {loadingCoupons ? (
              <View style={styles.modalLoading}>
                <Text style={{ color: colors.fontColorBase }}>加载中...</Text>
              </View>
            ) : availableCoupons.length === 0 ? (
              <View style={styles.modalLoading}>
                <Text style={{ color: colors.fontColorDisabled }}>暂无可用优惠券</Text>
              </View>
            ) : (
              <FlatList
                data={availableCoupons}
                keyExtractor={(item) => String(item.id)}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[styles.couponItem, { borderColor: colors.border }]}
                    onPress={() => handleSelectCoupon(item)}
                  >
                    <View style={styles.couponItemLeft}>
                      <Text style={[styles.couponItemAmount, { color: colors.primary }]}>
                        ¥{item.amount}
                      </Text>
                      <Text style={[styles.couponItemCondition, { color: colors.fontColorDisabled }]}>
                        满¥{item.minPoint}可用
                      </Text>
                    </View>
                    <View style={[styles.couponItemDivider, { backgroundColor: colors.border }]} />
                    <View style={styles.couponItemRight}>
                      <Text style={[styles.couponItemName, { color: colors.fontColorDark }]} numberOfLines={1}>
                        {item.name}
                      </Text>
                      <Text style={[styles.couponItemTime, { color: colors.fontColorLight }]}>
                        有效期至 {item.endTime}
                      </Text>
                      <Text style={[styles.couponItemUseType, { color: colors.fontColorLight }]}>
                        {item.useType === 0 ? '全场通用' : item.useType === 1 ? '指定分类可用' : '指定商品可用'}
                      </Text>
                    </View>
                  </TouchableOpacity>
                )}
                contentContainerStyle={styles.couponListContent}
              />
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loading: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  addressSection: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  addressContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  addressInfo: {
    flex: 1,
  },
  addressTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  addressName: {
    fontSize: 16,
    fontWeight: '600',
  },
  addressPhone: {
    fontSize: 14,
  },
  addressDetail: {
    fontSize: 13,
  },
  addressPlaceholder: {
    flex: 1,
    fontSize: 15,
  },
  section: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  orderInfo: {
    gap: 4,
  },
  orderId: {
    fontSize: 14,
  },
  orderItems: {
    fontSize: 12,
  },
  couponRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  couponSelected: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  couponSelectedText: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
  },
  clearCouponBtn: {
    padding: 2,
  },
  couponPlaceholder: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  couponPlaceholderText: {
    fontSize: 14,
    marginRight: 4,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '70%',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  modalLoading: {
    padding: 32,
    alignItems: 'center',
  },
  couponListContent: {
    paddingBottom: 16,
  },
  couponItem: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginTop: 12,
    borderRadius: 12,
    borderWidth: 1,
    overflow: 'hidden',
    backgroundColor: '#fff',
  },
  couponItemLeft: {
    width: 90,
    padding: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  couponItemAmount: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  couponItemCondition: {
    fontSize: 11,
    marginTop: 4,
  },
  couponItemDivider: {
    width: 1,
  },
  couponItemRight: {
    flex: 1,
    padding: 12,
    justifyContent: 'space-between',
  },
  couponItemName: {
    fontSize: 14,
    fontWeight: '500',
  },
  couponItemTime: {
    fontSize: 11,
    marginTop: 4,
  },
  couponItemUseType: {
    fontSize: 11,
    marginTop: 2,
  },
  footerDiscount: {
    fontSize: 12,
    marginRight: 8,
    textDecorationLine: 'line-through',
  },
  methodItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 12,
    marginBottom: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#eee',
  },
  methodLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  methodName: {
    fontSize: 15,
  },
  footer: {
    padding: 16,
    paddingBottom: 34,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  footerTotal: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginBottom: 12,
  },
  footerLabel: {
    fontSize: 14,
  },
  footerAmount: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  payButton: {
    paddingVertical: 14,
    borderRadius: 25,
    alignItems: 'center',
  },
  payButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
})
