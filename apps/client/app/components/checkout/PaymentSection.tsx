import React from 'react'
import { View, Text, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'

type PayMethod = 'wechat' | 'alipay' | 'stripe'

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

interface PaymentSectionProps {
  selectedMethod: PayMethod
  colors: typeof Colors.light
  onSelectMethod: (method: PayMethod) => void
}

export default function PaymentSection({ selectedMethod, colors, onSelectMethod }: PaymentSectionProps) {
  return (
    <View style={[styles.section, { backgroundColor: colors.background }]}>
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
        支付方式
      </Text>
      {PAY_METHODS.map((method) => (
        <TouchableOpacity
          key={method.id}
          style={[
            styles.methodItem,
            selectedMethod === method.id && {
              borderColor: colors.primary,
              borderWidth: 2,
            },
          ]}
          onPress={() => onSelectMethod(method.id)}
        >
          <View style={styles.methodLeft}>
            <IconSymbol
              name={method.icon as any}
              size={24}
              color={
                selectedMethod === method.id
                  ? colors.primary
                  : colors.fontColorBase
              }
            />
            <Text
              style={[
                styles.methodName,
                { color: colors.fontColorDark },
                selectedMethod === method.id && { color: colors.primary },
              ]}
            >
              {method.name}
            </Text>
          </View>
          <IconSymbol
            name={
              selectedMethod === method.id
                ? 'checkmark.circle.fill'
                : 'circle'
            }
            size={24}
            color={
              selectedMethod === method.id
                ? colors.primary
                : colors.fontColorDisabled
            }
          />
        </TouchableOpacity>
      ))}
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
  methodItem: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'space-between' as const,
    paddingVertical: 12,
    paddingHorizontal: 12,
    marginBottom: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#eee',
  },
  methodLeft: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: 12,
  },
  methodName: {
    fontSize: 15,
  },
}
