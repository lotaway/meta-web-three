import React from 'react'
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type AppColors = typeof Colors.light

const REFUND_REASONS = [
  { value: 1, label: '不想要了' },
  { value: 2, label: '商品损坏' },
  { value: 3, label: '与描述不符' },
  { value: 4, label: '质量问题' },
  { value: 5, label: '发错货' },
  { value: 6, label: '其他原因' },
]

interface ReasonSelectorProps {
  selectedReason: number | null
  setSelectedReason: (reason: number) => void
}

function ReasonButton({ reason, selected, colors, onPress }: { reason: { value: number; label: string }; selected: boolean; colors: AppColors; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.reasonBtn, {
        borderColor: selected ? colors.primary : colors.border,
        backgroundColor: selected ? colors.primary + '10' : 'transparent',
      }]}
      onPress={onPress}
    >
      <Text style={[styles.reasonBtnText, { color: selected ? colors.primary : colors.text }]}>
        {reason.label}
      </Text>
    </TouchableOpacity>
  )
}

export function ReasonSelector({ selectedReason, setSelectedReason }: ReasonSelectorProps) {
  const { t } = useTranslation()
  const colors = Colors[useColorScheme() ?? 'light']
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('refund.reason_title')}</Text>
      <View style={styles.reasonGrid}>
        {REFUND_REASONS.map(reason => (
          <ReasonButton key={reason.value} reason={reason} selected={selectedReason === reason.value} colors={colors} onPress={() => setSelectedReason(reason.value)} />
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
  reasonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  reasonBtn: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
  },
  reasonBtnText: {
    fontSize: 14,
  },
})
