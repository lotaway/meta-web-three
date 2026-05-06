import React from 'react'
import { View, Text, TouchableOpacity } from 'react-native'
import { Colors } from '@/constants/Colors'

interface AvailableCoupon {
  id: number
  name: string
  amount: number
}

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

interface SummarySectionProps {
  orderInfo: OrderInfo
  selectedCoupon: AvailableCoupon | null
  paying: boolean
  colors: typeof Colors.light
  onSubmit: () => void
}

export default function SummarySection({ orderInfo, selectedCoupon, paying, colors, onSubmit }: SummarySectionProps) {
  return (
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
        onPress={onSubmit}
        disabled={paying}
      >
        <Text style={styles.payButtonText}>
          {paying ? '支付中...' : `立即支付`}
        </Text>
      </TouchableOpacity>
    </View>
  )
}

const styles = {
  footer: {
    padding: 16,
    paddingBottom: 34,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  footerTotal: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'flex-end' as const,
    marginBottom: 12,
  },
  footerLabel: {
    fontSize: 14,
  },
  footerAmount: {
    fontSize: 22,
    fontWeight: 'bold' as const,
  },
  footerDiscount: {
    fontSize: 12,
    marginRight: 8,
    textDecorationLine: 'line-through' as const,
  },
  payButton: {
    paddingVertical: 14,
    borderRadius: 25,
    alignItems: 'center' as const,
  },
  payButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600' as const,
  },
}
