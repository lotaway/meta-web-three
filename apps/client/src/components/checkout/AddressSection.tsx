import React from 'react'
import { View, Text, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { Colors } from '@/constants/Colors'
import type { MemberReceiveAddressDTO } from '@/src/generated/api/models'

interface AddressSectionProps {
  selectedAddress: MemberReceiveAddressDTO | null
  colors: typeof Colors.light
  onPress: () => void
}

export default function AddressSection({ selectedAddress, colors, onPress }: AddressSectionProps) {
  return (
    <TouchableOpacity
      style={[styles.addressSection, { backgroundColor: colors.background }]}
      onPress={onPress}
    >
      {selectedAddress ? (
        <View style={styles.addressContent}>
          <IconSymbol name="location.fill" size={24} color={colors.primary} />
          <View style={styles.addressInfo}>
            <View style={styles.addressTop}>
              <Text style={[styles.addressName, { color: colors.fontColorDark }]}>
                {selectedAddress.name}
              </Text>
              <Text style={[styles.addressPhone, { color: colors.fontColorBase }]}>
                {selectedAddress.phoneNumber}
              </Text>
            </View>
            <Text style={[styles.addressDetail, { color: colors.fontColorLight }]}>
              {selectedAddress.province} {selectedAddress.city} {selectedAddress.region} {selectedAddress.detailAddress}
            </Text>
          </View>
          <IconSymbol name="chevron.right" size={20} color={colors.fontColorDisabled} />
        </View>
      ) : (
        <View style={styles.addressContent}>
          <IconSymbol name="location" size={24} color={colors.fontColorDisabled} />
          <Text style={[styles.addressPlaceholder, { color: colors.fontColorDisabled }]}>
            请选择收货地址
          </Text>
          <IconSymbol name="chevron.right" size={20} color={colors.fontColorDisabled} />
        </View>
      )}
    </TouchableOpacity>
  )
}

const styles = {
  addressSection: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  addressContent: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: 12,
  },
  addressInfo: {
    flex: 1,
  },
  addressTop: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    marginBottom: 4,
  },
  addressName: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  addressPhone: {
    fontSize: 14,
  },
  addressDetail: {
    fontSize: 13,
  },
  addressPlaceholder: {
    flex: 1,
    fontSize: 15,
  },
}
