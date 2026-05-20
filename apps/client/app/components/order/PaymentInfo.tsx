import React from 'react'
import { View, Text } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface PaymentInfoProps {
  totalAmount: number
  freightAmount: number
  couponAmount: number
  payAmount: number
}

export function PaymentInfo({
  totalAmount,
  freightAmount,
  couponAmount,
  payAmount,
}: PaymentInfoProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.amount_title')}</Text>
      <View style={styles.infoRow}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.product_amount')}</Text>
        <Text style={[styles.infoValue, { color: colors.text }]}>¥{totalAmount}</Text>
      </View>
      {freightAmount > 0 && (
        <View style={styles.infoRow}>
          <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.freight')}</Text>
          <Text style={[styles.infoValue, { color: colors.text }]}>¥{freightAmount}</Text>
        </View>
      )}
      {couponAmount > 0 && (
        <View style={styles.infoRow}>
          <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.coupon')}</Text>
          <Text style={[styles.infoValue, { color: '#FF3B30' }]}>-¥{couponAmount}</Text>
        </View>
      )}
      <View style={[styles.infoRow, styles.totalRow, { borderTopColor: colors.border }]}>
        <Text style={[styles.totalLabel, { color: colors.text }]}>{t('orders.pay_amount')}</Text>
        <Text style={[styles.totalValue, { color: colors.primary }]}>¥{payAmount}</Text>
      </View>
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
