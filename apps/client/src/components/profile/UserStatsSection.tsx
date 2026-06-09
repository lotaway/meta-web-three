import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import { useTranslation } from 'react-i18next'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

interface UserStatsSectionProps {
  user: { integration: number; growth: number; couponCount: number }
  colors: any
}

export default function UserStatsSection({ user, colors }: UserStatsSectionProps) {
  const { t } = useTranslation()
  const colorScheme = useColorScheme() ?? 'light'
  const themeColors = Colors[colorScheme]
  return (
    <View style={styles.statsSection}>
      <View style={styles.statItem}>
        <Text style={[styles.statNum, { color: colors.fontColorDark }]}>{user.integration}</Text>
        <Text style={[styles.statLabel, { color: colors.fontColorLight }]}>{t('profile.stats.integration')}</Text>
      </View>
      <View style={styles.statItem}>
        <Text style={[styles.statNum, { color: colors.fontColorDark }]}>{user.growth}</Text>
        <Text style={[styles.statLabel, { color: colors.fontColorLight }]}>{t('profile.stats.growth')}</Text>
      </View>
      <View style={styles.statItem}>
        <Text style={[styles.statNum, { color: colors.fontColorDark }]}>{user.couponCount}</Text>
        <Text style={[styles.statLabel, { color: colors.fontColorLight }]}>{t('profile.stats.coupon')}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  statsSection: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    paddingVertical: 20,
    marginHorizontal: 15,
    borderRadius: 8,
    marginTop: -20,
    elevation: 4,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
    zIndex: 10,
  },
  statItem: {
    alignItems: 'center',
  },
  statNum: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 12,
    marginTop: 4,
  },
})
