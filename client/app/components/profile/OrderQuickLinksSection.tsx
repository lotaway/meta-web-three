import React from 'react'
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import { IconSymbol } from '@/components/ui/IconSymbol'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface OrderQuickLinksSectionProps {
  colors: any
  isAuthenticated: boolean
  router: any
}

function requireAuth(authenticated: boolean, router: any, callback: () => void) {
  if (!authenticated) {
    router.push('/auth/login')
    return
  }
  callback()
}

export default function OrderQuickLinksSection({ colors, isAuthenticated, router }: OrderQuickLinksSectionProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const themeColors = Colors[colorScheme]
  const ORDER_STATUS_LINKS = [
    { label: t('profile.orders.all'), icon: 'list.bullet.rectangle', action: () => requireAuth(isAuthenticated, router, () => router.push('/orders')) },
    { label: t('profile.orders.unpaid'), icon: 'creditcard', action: () => requireAuth(isAuthenticated, router, () => router.push('/orders?status=unpaid')) },
    { label: t('profile.orders.undelivered'), icon: 'shippingbox', action: () => requireAuth(isAuthenticated, router, () => router.push('/orders?status=undelivered')) },
    { label: t('profile.orders.refund'), icon: 'arrow.counterclockwise', action: () => requireAuth(isAuthenticated, router, () => router.push('/orders?status=refund')) },
  ]

  return (
    <View style={styles.orderSection}>
      <View style={styles.orderHeader}>
        <Text style={[styles.orderTitle, { color: colors.fontColorDark }]}>{t('profile.orders.title')}</Text>
        <TouchableOpacity style={styles.seeAll} onPress={() => requireAuth(isAuthenticated, router, () => router.push('/orders'))}>
          <Text style={{ color: colors.fontColorLight }}>{t('profile.orders.see_all')}</Text>
          <IconSymbol name="chevron.right" size={12} color={colors.fontColorLight} />
        </TouchableOpacity>
      </View>
      <View style={styles.orderGrid}>
        {ORDER_STATUS_LINKS.map((item, index) => (
          <TouchableOpacity key={index} style={styles.orderItem} onPress={item.action}>
            <IconSymbol name={item.icon as any} size={28} color={colors.primary} />
            <Text style={[styles.orderLabel, { color: colors.fontColorDark }]}>{item.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  orderSection: {
    backgroundColor: '#fff',
    marginHorizontal: 15,
    marginTop: 15,
    borderRadius: 8,
    padding: 15,
  },
  orderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  orderTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  seeAll: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  orderGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  orderItem: {
    alignItems: 'center',
  },
  orderLabel: {
    fontSize: 12,
    marginTop: 8,
  },
})
