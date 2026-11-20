import React from 'react'
import { View, Text } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light

interface OrderInfoProps {
  order: any
}

function formatOrderDate(date: string) {
  return date ? new Date(date).toLocaleString() : '-'
}

function getPayTypeText(payType: number, t: Translate) {
  const payTypeTexts: Record<number, string> = {
    0: t('orders.pay_type_wechat'),
    1: t('orders.pay_type_alipay'),
    2: t('orders.pay_type_stripe'),
  }
  return payTypeTexts[payType] || '-'
}

function InfoRow({ label, value, colors }: { label: string; value: string; colors: AppColors }) {
  return (
    <View style={styles.infoRow}>
      <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.infoValue, { color: colors.text }]}>{value}</Text>
    </View>
  )
}

export function OrderInfo({ order }: OrderInfoProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.info_title')}</Text>
      <InfoRow label={t('orders.order_no')} value={order.orderSn || order.id} colors={colors} />
      <InfoRow label={t('orders.create_time')} value={formatOrderDate(order.createTime)} colors={colors} />
      <InfoRow label={t('orders.pay_time')} value={formatOrderDate(order.paymentTime)} colors={colors} />
      <InfoRow label={t('orders.pay_type')} value={getPayTypeText(order.payType, t)} colors={colors} />
      {!!order.note && <InfoRow label={t('orders.note')} value={order.note} colors={colors} />}
    </View>
  )
}

const styles = {
  section: {
    marginBottom: 10,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    paddingVertical: 8,
  },
  infoLabel: { fontSize: 14 },
  infoValue: { fontSize: 14 },
}
