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

function SelectionIcon({
  isSelected,
  colors,
}: {
  isSelected: boolean
  colors: typeof Colors.light
}) {
  return (
    <IconSymbol
      name={isSelected ? 'checkmark.circle.fill' : 'circle'}
      size={24}
      color={isSelected ? colors.primary : colors.fontColorDisabled}
    />
  )
}

function MethodLabel({ method, isSelected, colors }: {
  method: PayMethodOption; isSelected: boolean; colors: typeof Colors.light
}) {
  return (
    <View style={styles.methodLeft}>
      <IconSymbol
        name={method.icon as any}
        size={24}
        color={isSelected ? colors.primary : colors.fontColorBase}
      />
      <Text style={[styles.methodName, { color: colors.fontColorDark }, isSelected && { color: colors.primary }]}>
        {method.name}
      </Text>
    </View>
  )
}

function PayMethodItem({ method, selectedMethod, colors, onSelectMethod }: {
  method: PayMethodOption
  selectedMethod: PayMethod
  colors: typeof Colors.light
  onSelectMethod: (method: PayMethod) => void
}) {
  const isSelected = selectedMethod === method.id
  return (
    <TouchableOpacity
      style={[styles.methodItem, isSelected && { borderColor: colors.primary, borderWidth: 2 }]}
      onPress={() => onSelectMethod(method.id)}
    >
      <MethodLabel method={method} isSelected={isSelected} colors={colors} />
      <SelectionIcon isSelected={isSelected} colors={colors} />
    </TouchableOpacity>
  )
}

export default function PaymentSection({ selectedMethod, colors, onSelectMethod }: PaymentSectionProps) {
  return (
    <View style={[styles.section, { backgroundColor: colors.background }]}>
      <Text style={[styles.sectionTitle, { color: colors.fontColorDark }]}>
        支付方式
      </Text>
      {PAY_METHODS.map((method) => (
        <PayMethodItem
          key={method.id}
          method={method}
          selectedMethod={selectedMethod}
          colors={colors}
          onSelectMethod={onSelectMethod}
        />
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
