import React, { useState, useEffect, useCallback } from 'react'
import {
  View,
  Text,
  ScrollView,
  Alert,
  StyleSheet,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router, useLocalSearchParams, useFocusEffect } from 'expo-router'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { pay, type PayResult } from '@/lib/payment'
import { DEFAULT_USER_ID, orderApi, payApi, addressApi, couponApi } from '@/api/generated'
import type { ApiResponseLong, ApiResponseMapStringObject, ApiResponseMapStringString, OrderItemCreate, MemberReceiveAddressDTO } from '@/src/generated/api/models'
import { recommendationHooks } from '@/lib/api/graphql-hooks'
import AddressSection from '@/components/checkout/AddressSection'
import ProductListSection from '@/components/checkout/ProductListSection'
import CouponSection from '@/components/checkout/CouponSection'
import PaymentSection from '@/components/checkout/PaymentSection'
import SummarySection from '@/components/checkout/SummarySection'

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
        const defaultAddr = response.data.find((addr: MemberReceiveAddressDTO) => addr.defaultStatus === true)
        setSelectedAddress(defaultAddr ?? response.data[0])
      }
    } catch (error) {
      console.error('Failed to load address:', error)
    }
  }

  const loadAvailableCoupons = async () => {
    setLoadingCoupons(true)
    try {
      const response = await couponApi.listCoupons({
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

  const loadOrderBase = async () => {
    if (checkoutItems.length === 0) {
      throw new Error('未找到可结算商品')
    }
    const orderResponse = await createOrder(checkoutItems, selectedAddress?.id)
    if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
      throw new Error(orderResponse.message || '创建订单失败')
    }
    return orderResponse.data
  }

  const loadOrderDetail = async (orderId: number) => {
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
      items: detail?.items?.map((item: any) => ({
        name: item.productName ?? '',
        price: Number(item.unitPrice ?? 0),
        quantity: item.quantity ?? 0,
      })) ?? mapItemsForView(checkoutItems),
    })
  }

  const loadOrder = async () => {
    try {
      const orderId = await loadOrderBase()
      await loadOrderDetail(orderId)
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

  const createOrderWithCoupon = async (): Promise<number> => {
    const orderResponse = await createOrder(
      checkoutItems,
      selectedAddress?.id,
      selectedCoupon?.id,
      orderRemark
    )
    if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
      throw new Error(orderResponse.message || '创建订单失败')
    }
    return orderResponse.data
  }

  const processPaymentAndVerify = async (orderId: number): Promise<PayResult> => {
    const params = await getPaymentParams(orderId, selectedMethod)
    const result = await processPayment(selectedMethod, params)
    if (result.status === 'success') {
      if (!result.transactionId) {
        throw new Error('支付结果缺少交易凭证')
      }
      const verification = await verifyPayment(orderId, result.transactionId)
      const valid = verification.data?.valid === true
      if (!valid) {
        throw new Error('支付校验失败')
      }
      await markOrderPaid(orderId, selectedMethod)
    }
    return result
  }

  const handlePaymentWithOrder = async (finalAmount: number): Promise<PayResult> => {
    let currentOrderId = orderInfo!.orderId
    if (selectedCoupon) {
      currentOrderId = await createOrderWithCoupon()
    }
    return await processPaymentAndVerify(currentOrderId)
  }

  const processPayment = async (method: PayMethod, params: PaymentParams): Promise<PayResult> => {
    switch (method) {
      case 'wechat':
        if (!params) throw new Error('微信支付参数获取失败')
        return await pay('wechat', params as any)
      case 'alipay':
        if (!params?.orderString) throw new Error('支付宝支付参数获取失败')
        return await pay('alipay', params.orderString)
      case 'stripe':
        if (!params?.clientSecret) throw new Error('银行卡支付参数获取失败')
        return await pay('stripe', { clientSecret: params.clientSecret, returnURL: params.returnURL || 'app://payment' })
    }
  }

  const handlePayResult = (result: PayResult, orderId: number) => {
    switch (result.status) {
      case 'success':
        recommendationHooks.markPurchasedByProducts(
          DEFAULT_USER_ID,
          checkoutItems.map(item => item.id),
        ).catch(() => {})
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

  const handleOpenCouponSelector = () => {
    loadAvailableCoupons()
    setCouponModalVisible(true)
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
        <AddressSection
          selectedAddress={selectedAddress}
          colors={colors}
          onPress={() => router.push('/address/list?source=checkout')}
        />

        <ProductListSection orderInfo={orderInfo} colors={colors} />

        <CouponSection
          selectedCoupon={selectedCoupon}
          colors={colors}
          couponModalVisible={couponModalVisible}
          availableCoupons={availableCoupons}
          loadingCoupons={loadingCoupons}
          onOpenCouponSelector={handleOpenCouponSelector}
          onSelectCoupon={handleSelectCoupon}
          onClearCoupon={handleClearCoupon}
          onCloseModal={() => setCouponModalVisible(false)}
        />

        <PaymentSection
          selectedMethod={selectedMethod}
          colors={colors}
          onSelectMethod={setSelectedMethod}
        />
      </ScrollView>

      <SummarySection
        orderInfo={orderInfo}
        selectedCoupon={selectedCoupon}
        paying={paying}
        colors={colors}
        onSubmit={handlePay}
      />
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
})
