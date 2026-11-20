import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router, useLocalSearchParams, Stack } from 'expo-router';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useAuth } from '@/contexts/AuthContext';
import { createFlashOrder } from '@/adapters/flashAdapter';

function buildFlashOrderPayload(
  params: ReturnType<typeof useLocalSearchParams>,
  sessionId: number,
  productId: number,
  skuId: number,
  quantity: number,
  flashPrice: number,
) {
  return {
    sessionId,
    productId,
    skuId,
    productName: String(params.productName || ''),
    productPic: String(params.productPic || ''),
    quantity,
    flashPrice,
  };
}

function createSubmitHandler(token: string | null, payload: ReturnType<typeof buildFlashOrderPayload>, setSubmitting: (submitting: boolean) => void) {
  return async () => {
    if (!token) {
      Alert.alert('提示', '请先登录')
      router.push('/auth/login')
      return
    }
    setSubmitting(true)
    try {
      const { orderId } = await createFlashOrder(token, payload)
      router.replace(`/orders/${orderId}`)
    } catch (e: any) {
      Alert.alert('下单失败', e.message || '请重试')
    } finally {
      setSubmitting(false)
    }
  }
}

function OrderSummary({ themeColors, quantity, flashPrice, totalAmount }: {
  themeColors: any;
  quantity: number;
  flashPrice: number;
  totalAmount: number;
}) {
  return (
    <View style={[styles.summaryCard, { backgroundColor: '#fff' }]}>
      <Text style={[styles.sectionTitle, { color: themeColors.fontColorDark }]}>订单摘要</Text>
      <View style={styles.row}>
        <Text style={[styles.label, { color: themeColors.fontColorLight }]}>商品数量</Text>
        <Text style={[styles.value, { color: themeColors.fontColorDark }]}>{quantity} 件</Text>
      </View>
      <View style={styles.row}>
        <Text style={[styles.label, { color: themeColors.fontColorLight }]}>闪购单价</Text>
        <Text style={[styles.value, { color: '#ff2d2d', fontWeight: 'bold' }]}>￥{flashPrice}</Text>
      </View>
      <View style={[styles.row, styles.totalRow]}>
        <Text style={[styles.label, { color: themeColors.fontColorLight }]}>应付总额</Text>
        <Text style={[styles.totalAmount, { color: '#ff2d2d' }]}>￥{totalAmount}</Text>
      </View>
    </View>
  );
}

function SubmitButton({ submitting, onPress }: { submitting: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.submitBtn, submitting && { opacity: 0.6 }]}
      onPress={onPress}
      disabled={submitting}
    >
      {submitting ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <Text style={styles.submitText}>提交闪购订单</Text>
      )}
    </TouchableOpacity>
  );
}

interface FlashParams {
  sessionId: number
  productId: number
  skuId: number
  quantity: number
  flashPrice: number
  totalAmount: number
}

function parseFlashParams(params: ReturnType<typeof useLocalSearchParams>): FlashParams {
  const sessionId = Number(params.sessionId)
  const productId = Number(params.productId)
  const skuId = Number(params.skuId)
  const quantity = Number(params.quantity)
  const flashPrice = Number(params.flashPrice)
  return { sessionId, productId, skuId, quantity, flashPrice, totalAmount: flashPrice * quantity }
}

function FlashCheckoutView({ themeColors, quantity, flashPrice, totalAmount, submitting, onPress }: {
  themeColors: any;
  quantity: number;
  flashPrice: number;
  totalAmount: number;
  submitting: boolean;
  onPress: () => void;
}) {
  return (
    <SafeAreaView style={[styles.container, { backgroundColor: themeColors.background }]}>
      <Stack.Screen options={{ title: '闪购订单确认' }} />
      <View style={styles.content}>
        <OrderSummary themeColors={themeColors} quantity={quantity} flashPrice={flashPrice} totalAmount={totalAmount} />
        <SubmitButton submitting={submitting} onPress={onPress} />
      </View>
    </SafeAreaView>
  );
}

export default function FlashCheckoutScreen() {
  const { t } = { t: (k: string) => k };
  const colorScheme = useColorScheme() ?? 'light';
  const themeColors = Colors[colorScheme];
  const { token } = useAuth();
  const params = useLocalSearchParams();
  const [submitting, setSubmitting] = useState(false);
  const { sessionId, productId, skuId, quantity, flashPrice, totalAmount } = parseFlashParams(params)
  const payload = buildFlashOrderPayload(params, sessionId, productId, skuId, quantity, flashPrice)
  const handleSubmit = createSubmitHandler(token, payload, setSubmitting)
  return <FlashCheckoutView themeColors={themeColors} quantity={quantity} flashPrice={flashPrice} totalAmount={totalAmount} submitting={submitting} onPress={handleSubmit} />
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { flex: 1, padding: 15 },
  summaryCard: {
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 2,
  },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 15 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  label: { fontSize: 14 },
  value: { fontSize: 14 },
  totalRow: { borderTopWidth: 1, borderTopColor: '#f0f0f0', marginTop: 8, paddingTop: 12 },
  totalAmount: { fontSize: 20, fontWeight: 'bold' },
  submitBtn: {
    backgroundColor: '#ff2d2d',
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 30,
  },
  submitText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
});
