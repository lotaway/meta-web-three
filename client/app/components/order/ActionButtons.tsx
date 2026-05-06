import React from 'react'
import { View, Text, TouchableOpacity } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface ActionButtonsProps {
  status: number
  onPay: () => void
  onCancel: () => void
  onConfirmReceive: () => void
  onRefund: () => void
  onContactDelivery: () => void
  onReview: () => void
}

export function ActionButtons({
  status,
  onPay,
  onCancel,
  onConfirmReceive,
  onRefund,
  onContactDelivery,
  onReview,
}: ActionButtonsProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  const renderStatus0 = () => (
    <>
      <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onCancel}>
        <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.cancel')}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.bottomBtn, styles.payBottomBtn, { backgroundColor: colors.primary }]} onPress={onPay}>
        <Text style={styles.payBottomBtnText}>{t('orders.pay')}</Text>
      </TouchableOpacity>
    </>
  )

  const renderStatus1 = () => (
    <>
      <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onRefund}>
        <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onContactDelivery}>
        <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.contact_delivery')}</Text>
      </TouchableOpacity>
    </>
  )

  const renderStatus2 = () => (
    <>
      <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onRefund}>
        <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.bottomBtn, styles.confirmBottomBtn, { backgroundColor: colors.primary }]} onPress={onConfirmReceive}>
        <Text style={styles.confirmBottomBtnText}>{t('orders.confirm_receive')}</Text>
      </TouchableOpacity>
    </>
  )

  const renderStatus3 = () => (
    <>
      <TouchableOpacity style={[styles.bottomBtn, { borderColor: colors.border }]} onPress={onRefund}>
        <Text style={[styles.bottomBtnText, { color: colors.text }]}>{t('orders.refund')}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.bottomBtn, styles.confirmBottomBtn, { backgroundColor: colors.primary }]} onPress={onReview}>
        <Text style={styles.confirmBottomBtnText}>评价商品</Text>
      </TouchableOpacity>
    </>
  )

  const renderButtons = () => {
    switch (status) {
      case 0: return renderStatus0()
      case 1: return renderStatus1()
      case 2: return renderStatus2()
      case 3: return renderStatus3()
      default: return null
    }
  }

  return (
    <View style={[styles.bottomBar, { backgroundColor: colors.background, borderTopColor: colors.border }]}>
      {renderButtons()}
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
