import React from 'react'
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

const REFUND_TYPES = [
  { value: 1, label: '退款退货' },
  { value: 2, label: '仅退款' },
]

interface RefundTypeSelectorProps {
  refundType: number
  setRefundType: (type: number) => void
}

export function RefundTypeSelector({ refundType, setRefundType }: RefundTypeSelectorProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.type_title')}</Text>
      <View style={styles.typeRow}>
        {REFUND_TYPES.map(type => (
          <TouchableOpacity
            key={type.value}
            style={[
              styles.typeBtn,
              { borderColor: refundType === type.value ? colors.primary : colors.border },
              refundType === type.value && { backgroundColor: colors.primary + '10' },
            ]}
            onPress={() => setRefundType(type.value)}
          >
            <Text
              style={[
                styles.typeBtnText,
                { color: refundType === type.value ? colors.primary : colors.textSecondary },
              ]}
            >
              {type.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  section: {
    padding: 16,
    marginTop: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  typeRow: {
    flexDirection: 'row',
    gap: 12,
  },
  typeBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
  },
  typeBtnText: {
    fontSize: 14,
  },
})
