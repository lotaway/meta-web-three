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
import { pay, type PayResult, type PaySuccessResult } from '@/lib/payment'
import { DEFAULT_USER_ID, orderApi, payApi, addressApi, couponApi } from '@/api/generated'
import type { ApiResponseLong, ApiResponseMapStringObject, ApiResponseMapStringString, OrderItemCreate, MemberAddress } from '@/src/generated/api/models'
import { recommendationHooks } from '@/lib/api/graphql-hooks'
import AddressSection from '@/components/checkout/AddressSection'
import ProductListSection from '@/components/checkout/ProductListSection'
import CouponSection from '@/components/checkout/CouponSection'
import PaymentSection from '@/components/checkout/PaymentSection'
import SummarySection from '@/components/checkout/SummarySection'

type PayMethod = 'wechat' | 'alipay' | 'stripe'
type CheckoutColors = typeof Colors.light
type SetBoolean = React.Dispatch<React.SetStateAction<boolean>>

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

interface PaymentContext {
  checkoutItems: CheckoutItemParam[]
  selectedAddress: MemberAddress | null
  selectedCoupon: AvailableCoupon | null
  orderRemark: string
  selectedMethod: PayMethod
}

interface CheckoutState {
  orderInfo: OrderInfo | null
  setOrderInfo: React.Dispatch<React.SetStateAction<OrderInfo | null>>
  selectedMethod: PayMethod
  setSelectedMethod: React.Dispatch<React.SetStateAction<PayMethod>>
  paying: boolean
  setPaying: SetBoolean
  loading: boolean
  setLoading: SetBoolean
  selectedAddress: MemberAddress | null
  setSelectedAddress: React.Dispatch<React.SetStateAction<MemberAddress | null>>
  couponModalVisible: boolean
  setCouponModalVisible: SetBoolean
  availableCoupons: AvailableCoupon[]
  setAvailableCoupons: React.Dispatch<React.SetStateAction<AvailableCoupon[]>>
  selectedCoupon: AvailableCoupon | null
  setSelectedCoupon: React.Dispatch<React.SetStateAction<AvailableCoupon | null>>
  loadingCoupons: boolean
  setLoadingCoupons: SetBoolean
  orderRemark: string
  setOrderRemark: React.Dispatch<React.SetStateAction<string>>
}

interface CheckoutViewProps {
  colors: CheckoutColors
  orderInfo: OrderInfo
  checkoutItems: CheckoutItemParam[]
  selectedAddress: MemberAddress | null
  selectedMethod: PayMethod
  paying: boolean
  couponModalVisible: boolean
  availableCoupons: AvailableCoupon[]
  loadingCoupons: boolean
  selectedCoupon: AvailableCoupon | null
  onSelectAddress: () => void
  onSelectMethod: (method: PayMethod) => void
  onOpenCouponSelector: () => void
  onSelectCoupon: (coupon: AvailableCoupon) => void
  onClearCoupon: () => void
  onCloseCouponModal: () => void
  onPay: () => void
}

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
  return orderApi.create2({
    xUserId: DEFAULT_USER_ID,
    orderCreateRequest: {
      items: mapItemsToOrderCreate(items),
      memberReceiveAddressId: addressId,
      couponId: couponId,
      remark: remark,
    } as any,
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
    throw new Error(response.message || 'Failed to get payment parameters')
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

async function loadDefaultAddress(
  setSelectedAddress: React.Dispatch<React.SetStateAction<MemberAddress | null>>
): Promise<void> {
  try {
    const response = await addressApi.list3({ xUserId: DEFAULT_USER_ID })
    if (response.data && Array.isArray(response.data) && response.data.length > 0) {
      const defaultAddr = response.data.find((addr: MemberAddress) => addr.defaultStatus === true)
      setSelectedAddress(defaultAddr ?? response.data[0])
    }
  } catch (error) {
    console.error('Failed to load address:', error)
    Alert.alert('Error', 'Failed to load the address.')
  }
}

async function loadAvailableCoupons(
  setLoadingCoupons: SetBoolean,
  setAvailableCoupons: React.Dispatch<React.SetStateAction<AvailableCoupon[]>>
): Promise<void> {
  setLoadingCoupons(true)
  try {
    const response = await couponApi.listCoupons({ useStatus: 0 })
    if (response.data && Array.isArray(response.data)) {
      setAvailableCoupons(response.data as any)
    }
  } catch (error) {
    console.error('Failed to load coupons:', error)
    Alert.alert('Error', 'Failed to load coupons.')
  } finally {
    setLoadingCoupons(false)
  }
}

function selectCoupon(
  coupon: AvailableCoupon,
  setSelectedCoupon: React.Dispatch<React.SetStateAction<AvailableCoupon | null>>,
  setCouponModalVisible: SetBoolean
) {
  setSelectedCoupon(coupon)
  setCouponModalVisible(false)
}

function openCouponSelector(
  setLoadingCoupons: SetBoolean,
  setAvailableCoupons: React.Dispatch<React.SetStateAction<AvailableCoupon[]>>,
  setCouponModalVisible: SetBoolean
) {
  loadAvailableCoupons(setLoadingCoupons, setAvailableCoupons)
  setCouponModalVisible(true)
}

async function loadOrderBase(
  checkoutItems: CheckoutItemParam[],
  selectedAddress: MemberAddress | null
): Promise<number> {
  if (checkoutItems.length === 0) {
    throw new Error('No checkout items found')
  }
  const orderResponse = await createOrder(checkoutItems, selectedAddress?.id)
  if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
    throw new Error(orderResponse.message || 'Failed to create order')
  }
  return orderResponse.data
}

function buildOrderInfo(
  orderId: number,
  detail: any,
  fallbackItems: CheckoutItemParam[]
): OrderInfo {
  return {
    orderId,
    amount: Number(detail?.order?.orderAmount ?? 0),
    couponDiscount: 0,
    payAmount: Number(detail?.order?.orderAmount ?? 0),
    items: detail?.items?.map((item: any) => ({
      name: item.productName ?? '',
      price: Number(item.unitPrice ?? 0),
      quantity: item.quantity ?? 0,
    })) ?? mapItemsForView(fallbackItems),
  }
}

async function loadOrderDetail(
  orderId: number,
  checkoutItems: CheckoutItemParam[],
  setOrderInfo: React.Dispatch<React.SetStateAction<OrderInfo | null>>
): Promise<void> {
  const detailResponse = await orderApi.detail1({
    xUserId: DEFAULT_USER_ID,
    id: orderId,
  })
  setOrderInfo(buildOrderInfo(orderId, detailResponse.data, checkoutItems))
}

async function loadOrder(
  checkoutItems: CheckoutItemParam[],
  selectedAddress: MemberAddress | null,
  setOrderInfo: React.Dispatch<React.SetStateAction<OrderInfo | null>>,
  setLoading: SetBoolean
): Promise<void> {
  try {
    const orderId = await loadOrderBase(checkoutItems, selectedAddress)
    await loadOrderDetail(orderId, checkoutItems, setOrderInfo)
  } catch (error) {
    Alert.alert('错误', '加载订单失败')
    router.back()
  } finally {
    setLoading(false)
  }
}

async function createOrderWithCoupon(context: PaymentContext): Promise<number> {
  const orderResponse = await createOrder(
    context.checkoutItems,
    context.selectedAddress?.id,
    context.selectedCoupon?.id,
    context.orderRemark
  )
  if (!isSuccessCode(orderResponse.code) || !orderResponse.data) {
    throw new Error(orderResponse.message || 'Failed to create order')
  }
  return orderResponse.data
}

async function processPayment(method: PayMethod, params: PaymentParams): Promise<PayResult> {
  switch (method) {
    case 'wechat':
      if (!params) throw new Error('Failed to get WeChat payment parameters')
      return await pay('wechat', params as any)
    case 'alipay':
      if (!params?.orderString) throw new Error('Failed to get Alipay payment parameters')
      return await pay('alipay', params.orderString)
    case 'stripe':
      if (!params?.clientSecret) throw new Error('Failed to get card payment parameters')
      return await pay('stripe', { clientSecret: params.clientSecret, returnURL: params.returnURL || 'app://payment' })
  }
}

async function verifySuccessfulPayment(orderId: number, result: PaySuccessResult): Promise<void> {
  if (!result.transactionId) {
    throw new Error('Payment result is missing a transaction voucher')
  }
  const verification = await verifyPayment(orderId, result.transactionId)
  const valid = verification.data?.valid === true
  if (!valid) {
    throw new Error('Payment verification failed')
  }
}

async function processPaymentAndVerify(
  orderId: number,
  selectedMethod: PayMethod
): Promise<PayResult> {
  const params = await getPaymentParams(orderId, selectedMethod)
  const result = await processPayment(selectedMethod, params)
  if (result.status === 'success') {
    await verifySuccessfulPayment(orderId, result)
  }
  return result
}

async function handlePaymentWithOrder(orderInfo: OrderInfo, context: PaymentContext): Promise<PayResult> {
  let currentOrderId = orderInfo.orderId
  if (context.selectedCoupon) {
    currentOrderId = await createOrderWithCoupon(context)
  }
  return await processPaymentAndVerify(currentOrderId, context.selectedMethod)
}

function markPurchased(checkoutItems: CheckoutItemParam[]) {
  recommendationHooks.markPurchasedByProducts(
    DEFAULT_USER_ID,
    checkoutItems.map(item => item.id),
  ).catch(() => {})
}

function handlePayResult(
  result: PayResult,
  orderId: number,
  checkoutItems: CheckoutItemParam[]
) {
  switch (result.status) {
    case 'success':
      markPurchased(checkoutItems)
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

async function handlePay(
  orderInfo: OrderInfo | null,
  selectedCoupon: AvailableCoupon | null,
  paying: boolean,
  setPaying: SetBoolean,
  context: PaymentContext
): Promise<void> {
  if (!orderInfo || paying) return
  setPaying(true)
  try {
    const finalAmount = orderInfo.amount - (selectedCoupon?.amount ?? 0)
    const payResult = await handlePaymentWithOrder(orderInfo, context)
    handlePayResult(payResult, orderInfo.orderId, context.checkoutItems)
  } catch (error: any) {
    Alert.alert('支付异常', error.message || '请稍后重试')
  } finally {
    setPaying(false)
  }
}

function createCheckoutActions(state: CheckoutState, checkoutItems: CheckoutItemParam[]) {
  const context: PaymentContext = {
    checkoutItems,
    selectedAddress: state.selectedAddress,
    selectedCoupon: state.selectedCoupon,
    orderRemark: state.orderRemark,
    selectedMethod: state.selectedMethod,
  }
  return {
    onSelectMethod: state.setSelectedMethod,
    onSelectCoupon: (coupon: AvailableCoupon) =>
      selectCoupon(coupon, state.setSelectedCoupon, state.setCouponModalVisible),
    onClearCoupon: () => state.setSelectedCoupon(null),
    onOpenCouponSelector: () =>
      openCouponSelector(state.setLoadingCoupons, state.setAvailableCoupons, state.setCouponModalVisible),
    onCloseCouponModal: () => state.setCouponModalVisible(false),
    onPay: () => handlePay(state.orderInfo, state.selectedCoupon, state.paying, state.setPaying, context),
  }
}

function useCheckoutState(): CheckoutState {
  const [orderInfo, setOrderInfo] = useState<OrderInfo | null>(null)
  const [selectedMethod, setSelectedMethod] = useState<PayMethod>('wechat')
  const [paying, setPaying] = useState(false)
  const [loading, setLoading] = useState(true)
  const [selectedAddress, setSelectedAddress] = useState<MemberAddress | null>(null)
  const [couponModalVisible, setCouponModalVisible] = useState(false)
  const [availableCoupons, setAvailableCoupons] = useState<AvailableCoupon[]>([])
  const [selectedCoupon, setSelectedCoupon] = useState<AvailableCoupon | null>(null)
  const [loadingCoupons, setLoadingCoupons] = useState(false)
  const [orderRemark, setOrderRemark] = useState('')
  return {
    orderInfo, setOrderInfo, selectedMethod, setSelectedMethod, paying, setPaying,
    loading, setLoading, selectedAddress, setSelectedAddress, couponModalVisible,
    setCouponModalVisible, availableCoupons, setAvailableCoupons, selectedCoupon,
    setSelectedCoupon, loadingCoupons, setLoadingCoupons, orderRemark, setOrderRemark,
  }
}

function useCheckoutController() {
  const { items: itemsParam } = useLocalSearchParams<{ items?: string }>()
  const colors = Colors[useColorScheme() ?? 'light']
  const checkoutItems = parseCheckoutItems(itemsParam)
  const state = useCheckoutState()
  useEffect(() => {
    loadOrder(checkoutItems, state.selectedAddress, state.setOrderInfo, state.setLoading)
  }, [])
  useFocusEffect(useCallback(() => {
    loadDefaultAddress(state.setSelectedAddress)
  }, []))
  const actions = createCheckoutActions(state, checkoutItems)
  return { colors, checkoutItems, state, actions }
}

function CouponArea(props: CheckoutViewProps) {
  return (
    <CouponSection
      selectedCoupon={props.selectedCoupon}
      colors={props.colors}
      couponModalVisible={props.couponModalVisible}
      availableCoupons={props.availableCoupons}
      loadingCoupons={props.loadingCoupons}
      onOpenCouponSelector={props.onOpenCouponSelector}
      onSelectCoupon={props.onSelectCoupon}
      onClearCoupon={props.onClearCoupon}
      onCloseModal={props.onCloseCouponModal}
    />
  )
}

function CheckoutContent(props: CheckoutViewProps) {
  return (
    <ScrollView style={styles.content}>
      <AddressSection
        selectedAddress={props.selectedAddress}
        colors={props.colors}
        onPress={props.onSelectAddress}
      />
      <ProductListSection orderInfo={props.orderInfo} colors={props.colors} />
      <CouponArea {...props} />
      <PaymentSection
        selectedMethod={props.selectedMethod}
        colors={props.colors}
        onSelectMethod={props.onSelectMethod}
      />
    </ScrollView>
  )
}

function CheckoutView(props: CheckoutViewProps) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: props.colors.background }]}>
      <CheckoutContent {...props} />
      <SummarySection
        orderInfo={props.orderInfo}
        selectedCoupon={props.selectedCoupon}
        paying={props.paying}
        colors={props.colors}
        onSubmit={props.onPay}
      />
    </SafeAreaView>
  )
}

function CheckoutMessage({ colors, message }: { colors: CheckoutColors; message: string }) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.loading}>
        <Text style={{ color: colors.fontColorBase }}>{message}</Text>
      </View>
    </SafeAreaView>
  )
}

export default function CheckoutScreen() {
  const { colors, checkoutItems, state, actions } = useCheckoutController()
  if (state.loading) {
    return <CheckoutMessage colors={colors} message="加载中..." />
  }
  if (!state.orderInfo) {
    return <CheckoutMessage colors={colors} message="订单不存在" />
  }
  return (
    <CheckoutView
      {...state}
      {...actions}
      colors={colors}
      checkoutItems={checkoutItems}
      orderInfo={state.orderInfo}
      onSelectAddress={() => router.push('/address/list?source=checkout')}
    />
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
