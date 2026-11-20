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

function AddressHeader({ colors, title }: { colors: any; title: string }) {
  return (
    <View style={styles.addressHeader}>
      <IconSymbol name="location.fill" size={18} color={colors.primary} />
      <Text style={[styles.sectionTitle, { color: colors.text }]}>{title}</Text>
    </View>
  )
}

function AddressContent({
  receiverName,
  receiverPhone,
  receiverProvince,
  receiverCity,
  receiverRegion,
  receiverDetailAddress,
  colors,
}: AddressInfoProps & { colors: any }) {
  return (
    <View style={styles.addressContent}>
      <Text style={[styles.addressName, { color: colors.text }]}>
        {receiverName} {receiverPhone}
      </Text>
      <Text style={[styles.addressDetail, { color: colors.textSecondary }]}>
        {receiverProvince}{receiverCity}{receiverRegion}{receiverDetailAddress}
      </Text>
    </View>
  )
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
      <AddressHeader colors={colors} title={t('orders.address_title')} />
      <AddressContent
        receiverName={receiverName}
        receiverPhone={receiverPhone}
        receiverProvince={receiverProvince}
        receiverCity={receiverCity}
        receiverRegion={receiverRegion}
        receiverDetailAddress={receiverDetailAddress}
        colors={colors}
      />
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
