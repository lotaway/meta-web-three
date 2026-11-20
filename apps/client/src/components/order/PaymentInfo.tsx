import React from 'react'
import { View, Text } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type AppColors = typeof Colors.light

interface PaymentInfoProps {
  totalAmount: number
  freightAmount: number
  couponAmount: number
  payAmount: number
}

function AmountRow({ label, value, colors, valueColor }: {
  label: string
  value: string
  colors: AppColors
  valueColor?: string
}) {
  return (
    <View style={styles.infoRow}>
      <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.infoValue, { color: valueColor || colors.text }]}>{value}</Text>
    </View>
  )
}

function TotalRow({ label, value, colors }: { label: string; value: string; colors: AppColors }) {
  return (
    <View style={[styles.infoRow, styles.totalRow, { borderTopColor: colors.border }]}>
      <Text style={[styles.totalLabel, { color: colors.text }]}>{label}</Text>
      <Text style={[styles.totalValue, { color: colors.primary }]}>{value}</Text>
    </View>
  )
}

export function PaymentInfo({ totalAmount, freightAmount, couponAmount, payAmount }: PaymentInfoProps) {
  const { t } = useTranslation()
  const colors = Colors[useColorScheme() ?? 'light']
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.amount_title')}</Text>
      <AmountRow label={t('orders.product_amount')} value={`¥${totalAmount}`} colors={colors} />
      {freightAmount > 0 && <AmountRow label={t('orders.freight')} value={`¥${freightAmount}`} colors={colors} />}
      {couponAmount > 0 && <AmountRow label={t('orders.coupon')} value={`-¥${couponAmount}`} colors={colors} valueColor="#FF3B30" />}
      <TotalRow label={t('orders.pay_amount')} value={`¥${payAmount}`} colors={colors} />
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
  totalRow: {
    paddingTop: 12,
    marginTop: 4,
    borderTopWidth: 1,
  },
  totalLabel: { fontSize: 14, fontWeight: '600' as const },
  totalValue: { fontSize: 18, fontWeight: 'bold' as const },
}
