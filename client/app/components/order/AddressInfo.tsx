import React from 'react'
import { View, Text } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'
import { IconSymbol } from '@/components/ui/IconSymbol'

interface AddressInfoProps {
  receiverName: string
  receiverPhone: string
  receiverProvince: string
  receiverCity: string
  receiverRegion: string
  receiverDetailAddress: string
}

export function AddressInfo({
  receiverName,
  receiverPhone,
  receiverProvince,
  receiverCity,
  receiverRegion,
  receiverDetailAddress,
}: AddressInfoProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const colors = Colors[colorScheme]

  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <View style={styles.addressHeader}>
        <IconSymbol name="location.fill" size={18} color={colors.primary} />
        <Text style={[styles.sectionTitle, { color: colors.text }]}>{t('orders.address_title')}</Text>
      </View>
      <View style={styles.addressContent}>
        <Text style={[styles.addressName, { color: colors.text }]}>
          {receiverName} {receiverPhone}
        </Text>
        <Text style={[styles.addressDetail, { color: colors.textSecondary }]}>
          {receiverProvince}{receiverCity}{receiverRegion}{receiverDetailAddress}
        </Text>
      </View>
    </View>
  )
}

const styles = {
  section: {
    marginBottom: 10,
    padding: 16,
  },
  addressHeader: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: 8,
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  addressContent: { paddingLeft: 26 },
  addressName: { fontSize: 16, fontWeight: '500' as const },
  addressDetail: { fontSize: 14, marginTop: 4, lineHeight: 20 },
}
