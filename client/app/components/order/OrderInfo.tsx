import React from 'react'
import { View, Text } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface OrderInfoProps {
  order: any
}

export function OrderInfo({ order }: OrderInfoProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  const formatDate = (date: string) => {
    return date ? new Date(date).toLocaleString() : '-'
  }

  const getPayTypeText = (payType: number) => {
    if (payType === 0) return t('orders.pay_type_wechat')
    if (payType === 1) return t('orders.pay_type_alipay')
    if (payType === 2) return t('orders.pay_type_stripe')
    return '-'
  }

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.info_title')}</Text>
      <View style={styles.infoRow}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.order_no')}</Text>
        <Text style={[styles.infoValue, { color: colors.text }]}>{order.orderSn || order.id}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.create_time')}</Text>
        <Text style={[styles.infoValue, { color: colors.text }]}>{formatDate(order.createTime)}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.pay_time')}</Text>
        <Text style={[styles.infoValue, { color: colors.text }]}>{formatDate(order.paymentTime)}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.pay_type')}</Text>
        <Text style={[styles.infoValue, { color: colors.text }]}>{getPayTypeText(order.payType)}</Text>
      </View>
      {order.note && (
        <View style={styles.infoRow}>
          <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{t('orders.note')}</Text>
          <Text style={[styles.infoValue, { color: colors.text }]}>{order.note}</Text>
        </View>
      )}
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
