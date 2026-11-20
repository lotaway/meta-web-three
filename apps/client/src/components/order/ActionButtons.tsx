import React from 'react'
import { View, Text, TouchableOpacity } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type Translate = ReturnType<typeof useTranslation>['t']
type AppColors = typeof Colors.light

interface ActionButtonsProps {
  status: number
  orderId: number
  onPay: () => void
  onCancel: () => void
  onConfirmReceive: () => void
  onRefund: () => void
  onContactDelivery: () => void
  onReview: () => void
  onContactService?: () => void
}

interface StatusButtonsProps {
  status: number
  colors: AppColors
  t: Translate
  onPay: () => void
  onCancel: () => void
  onConfirmReceive: () => void
  onRefund: () => void
  onContactDelivery: () => void
  onReview: () => void
}

function BorderButton({ label, colors, onPress }: { label: string; colors: AppColors; onPress?: () => void }) {
  return (
    <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onPress}>
      <Text style={[styles.bottomBtnText, { color: colors.text }]}>{label}</Text>
    </TouchableOpacity>
  )
}

function PayButton({ label, colors, onPress }: { label: string; colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity style={[styles.bottomBtn, styles.payBottomBtn, { backgroundColor: colors.primary }]} onPress={onPress}>
      <Text style={styles.payBottomBtnText}>{label}</Text>
    </TouchableOpacity>
  )
}

function ConfirmButton({ label, colors, onPress }: { label: string; colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity style={[styles.bottomBtn, styles.confirmBottomBtn, { backgroundColor: colors.primary }]} onPress={onPress}>
      <Text style={styles.confirmBottomBtnText}>{label}</Text>
    </TouchableOpacity>
  )
}

function StatusButtons({ status, colors, t, onPay, onCancel, onConfirmReceive, onRefund, onContactDelivery, onReview }: StatusButtonsProps) {
  const refund = <BorderButton label={t('orders.refund')} colors={colors} onPress={onRefund} />
  switch (status) {
    case 0:
      return <><BorderButton label={t('orders.cancel')} colors={colors} onPress={onCancel} /><PayButton label={t('orders.pay')} colors={colors} onPress={onPay} /></>
    case 1:
      return <>{refund}<BorderButton label={t('orders.contact_delivery')} colors={colors} onPress={onContactDelivery} /></>
    case 2:
      return <>{refund}<ConfirmButton label={t('orders.confirm_receive')} colors={colors} onPress={onConfirmReceive} /></>
    case 3:
      return <>{refund}<ConfirmButton label="评价商品" colors={colors} onPress={onReview} /></>
    default:
      return null
  }
}

export function ActionButtons({ status, onPay, onCancel, onConfirmReceive, onRefund, onContactDelivery, onReview, onContactService }: ActionButtonsProps) {
  const { t } = useTranslation()
  const colors = Colors[useColorScheme() ?? 'light']
  return (
    <View style={[styles.bottomBar, { backgroundColor: colors.background, borderTopColor: colors.border }]}>
      <BorderButton label="客服" colors={colors} onPress={onContactService} />
      <StatusButtons status={status} colors={colors} t={t} onPay={onPay} onCancel={onCancel} onConfirmReceive={onConfirmReceive} onRefund={onRefund} onContactDelivery={onContactDelivery} onReview={onReview} />
    </View>
  )
}

const styles = {
  bottomBar: {
    flexDirection: 'row' as const,
    justifyContent: 'flex-end' as const,
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
  payBottomBtn: { borderWidth: 0 },
  payBottomBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' as const },
  confirmBottomBtn: { borderWidth: 0 },
  confirmBottomBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' as const },
}
