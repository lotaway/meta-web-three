import React from 'react'
import { View, Text } from 'react-native'
import { Colors } from '@/constants/Colors'

interface OrderItem {
  name: string
  price: number
  quantity: number
}

interface ProductListSectionProps {
  orderInfo: {
    orderId: number
    items: OrderItem[]
  }
  colors: typeof Colors.light
}

export default function ProductListSection({ orderInfo, colors }: ProductListSectionProps) {
  return (
    <View style={[styles.section, { backgroundColor: colors.background }]}>
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
        订单信息
      </Text>
      <View style={styles.orderInfo}>
        <Text style={[styles.orderId, { color: colors.fontColorBase }]}>
          订单号: {orderInfo.orderId}
        </Text>
        <Text style={[styles.orderItems, { color: colors.fontColorLight }]}>
          {orderInfo.items.map((item) => `${item.name} x${item.quantity}`).join(', ')}
        </Text>
      </View>
    </View>
  )
}

const styles = {
  section: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    marginBottom: 12,
  },
  orderInfo: {
    gap: 4,
  },
  orderId: {
    fontSize: 14,
  },
  orderItems: {
    fontSize: 12,
  },
}
